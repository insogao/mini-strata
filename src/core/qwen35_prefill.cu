// src/core/qwen35_prefill.cu - the Qwen35 prompt layer through Strata's original expert pipeline.
//
// See include/strata/core/qwen35_prefill.hpp.  The sequence mirrors src/prefill/prefill.cpp's MMQ branch:
// group by expert -> resident from the ExpertCache device slot / cold through a bounded pinned ring on the
// copy stream -> mmq::gather_native into the group buffer -> mmq::Product gate/up -> swiglu -> hidden
// quantizer -> mmq::Product down -> weighted combine.  The buffer sizes follow the model descriptor
// (n_embd 2048, n_ff 512, k 8) instead of the Flash-Next constants; the Qwen attention/GDN/shared expert/KV
// are the mature graph's and are not touched here.
#include "strata/core/qwen35_prefill.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/cold_stager.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
namespace {

constexpr int64_t kGroup = 16;    // experts per MMQ group (as prefill.cpp)
constexpr size_t kTail = 4096;    // the zeroed tail after the group's last expert (MMQ reads past it)
constexpr int kThreads = 256;

// ---- host routing fingerprint (diagnostic only, exact value "1", default off) ----
// Reads ONLY the ids/weights already staged on the host and the host residency table; no CUDA call, no order
// or scheduling change.  FNV-1a 64 over raw bytes: ids in (token, rank) order, weights as their raw float bits
// in the same order, plus per-call distinct/group/resident/streamed counts and the loan pair fingerprint.
constexpr uint64_t kFnvSeed = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
uint64_t fnv_bytes(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= kFnvPrime; }
    return h;
}
bool route_fp_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_ROUTE_FP");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
double ms_since(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
/// Exact-value gate for the optional per-call host phase accumulators (diagnostic; no CUDA events/syncs).
bool prefill_phases_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_PREFILL_PHASES");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the bounded per-request group-gather counters (diagnostic, default off).
bool gather_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_GATHER_STATS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the bounded real-weight byte check of the group gather (diagnostic, default off;
/// one bounded check per distinct (gu_type, d_type) actually taking the fast path in the allocation epoch,
/// with exact-size scratch, one explicit stream sync and four blocking D2H copies per check).
bool gather_check_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_GATHER_CHECK");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the sparse producer summary at close (diagnostic, default off).
bool stager_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STAGER_STATS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the producer's bounded host-job byte check: one blob per distinct (gu_type, d_type)
/// is compared against a synchronous copy_blobs reference (diagnostic, default off).
bool stager_check_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STAGER_CHECK");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the controlled negative: the (ring+1)-th staging copy's producer call is forced to
/// fail so the opt-in error path is exercised after the issuer has issued and the host ring has wrapped
/// (diagnostic, default off; while on, every source copy is routed through the counter).
bool stager_fail_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STAGER_FAIL");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the bounded cold-page hint (default off): while on, the current layer's transient
/// (GPU-miss) experts are handed to the source's advise_pairs before the producer starts, so the OS can start
/// the three mapped role slices early.  On Windows this is the file source's PrefetchVirtualMemory opt-in; off
/// adds no range building, no log, no API call, no allocation and no sync.
bool cold_hint_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_COLD_HINT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the write-combined staging ring (default off; exact "1" only): the Qwen prompt path
/// asks the stager for WC payload buffers (cudaHostAllocWriteCombined + a hardware store fence before ready).
/// It is refused in combination with STRATA_QWEN35_STAGER_CHECK (that check memcmp's the payload on the CPU);
/// unsupported platforms and flag-verification failures are refused by the stager's init.
bool stager_wc_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STAGER_WC");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the opt-in count-descending expert row layout (default off): only the exact value
/// "count_desc" enables it; unset or any other value keeps the original ascending-by-expert-ID layout.
bool expert_order_count_desc() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_EXPERT_ORDER");
        return v != nullptr && std::strcmp(v, "count_desc") == 0;
    }();
    return on;
}
/// Exact-value gate for the bounded per-call GPU event pairs (diagnostic, default off; exact "1").  Off: no
/// event objects, no records, no elapsed-time calls and no new IO anywhere.
bool gpu_events_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_GPU_EVENTS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Exact-value gate for the bounded per-layer routing-row histogram (host only, default off; exact "1").
/// Summarized once at close, never per call.
bool rowhist_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_ROWHIST");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// The one-shot single-layer capture gate (default off): STRATA_QWEN35_LAYER_CAPTURE is an existing directory;
/// only a non-empty value enables it, and only for the first T==2048 call of layer 16.
const char* layer_capture_dir() {
    static const char* v = [] {
        const char* e = std::getenv("STRATA_QWEN35_LAYER_CAPTURE");
        return (e != nullptr && e[0] != '\0') ? e : nullptr;
    }();
    return v;
}
/// POC fatal fault strategy: the device drain could not be confirmed at a point that is about to refill the
/// borrowed tail or release the producer's pinned ring.  One logged best-effort fallback sync already ran and
/// failed, so continuing would risk refilling/reusing memory an in-flight DMA still reads.  Terminate the
/// process non-zero WITHOUT the refill/release/destructors (no abort dialog, no false success); the OS
/// reclaims everything.  This is a driver-fault stop, not a recovery.
[[noreturn]] void fatal_drain_unconfirmed(const char* where, cudaError_t fb) {
    std::fprintf(stderr, "strata qwen35: FATAL: the device drain could not be confirmed at %s (%s); the prompt "
                         "loan is NOT refilled and the producer ring/workspace are NOT released\n",
                 where, cudaGetErrorString(fb));
    std::fflush(stderr);
    std::_Exit(1);
}
/// Exact-value gate for the p512 route-boundary probes (diagnostic, default off; needs
/// STRATA_QWEN35_ROUTE_PROBE_DIR): at the SECOND T==128 prompt ubatch of layer 16 and layer 17 it copies three
/// bounded arrays to host - A: layer16 expert input (attn_post_norm), B: layer16 combine output, C: layer17
/// expert input - hashes them, prints a record line and writes raw .f32 files.  Nothing runs when unset.
bool route_probe_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_ROUTE_PROBE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Adds host wall to `acc` on destruction when enabled; safe around early `continue` paths.
struct PhaseScope {
    bool on = false;
    std::chrono::steady_clock::time_point t0{};
    double* acc = nullptr;
    PhaseScope(bool enabled, double* a) : on(enabled), acc(a) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~PhaseScope() { if (on && acc != nullptr) *acc += ms_since(t0); }
};
struct RouteFpRec {
    int64_t call = 0, layer = 0, T = 0, k = 0;
    int64_t distinct = 0, groups = 0, resident = 0, streamed = 0;
    uint64_t ids_hash = 0, weights_hash = 0;
};
// group-gather byte-check registry (STRATA_QWEN35_GATHER_CHECK=1, diagnostic): one entry per distinct
// (gu_type, d_type) whose first fast group was re-gathered and compared byte for byte (written extents).
struct GatherChkPair {
    int32_t gu_type = -1, d_type = -1;
    int64_t layer = -1, j0 = -1;
    int32_t ngx = 0;
    size_t gu_slot_bytes = 0, d_slot_bytes = 0, gu_bytes = 0, d_bytes = 0;
};
constexpr int kGatherChkMax = 8;   // this model has at most 4 distinct (gu_type, d_type) layer-format pairs
/// Frees the byte-check scratch on every exit path, including the failure returns below.
struct GatherChkScratch {
    void* gu = nullptr;
    void* d = nullptr;
    ~GatherChkScratch() {
        if (gu != nullptr) cudaFree(gu);
        if (d != nullptr) cudaFree(d);
    }
};
// producer byte-check registry (STRATA_QWEN35_STAGER_CHECK=1): one host blob per distinct (gu_type, d_type)
// compared against a synchronous copy_blobs reference.
struct StagerChkPair {
    int32_t gu_type = -1, d_type = -1;
    int64_t layer = -1;
    int32_t expert = -1;
    size_t bytes = 0;
};
constexpr int kStagerChkMax = 8;
/// Controlled negative source (STRATA_QWEN35_STAGER_FAIL=1, default off): delegates to the real source except
/// that its first copying call returns false - the original producer's source-false branch, nothing else.
struct StagerFaultSource final : strata::core::ExpertSource {
    strata::core::ExpertSource* real = nullptr;
    // Atomic: with the opt-in two producers both may copy through this wrapper concurrently (STAGER_FAIL only).
    // The threshold counts total copy calls; the fetch_add order does not define which job fails.
    std::atomic<int> calls{0};
    int fail_at = 0;
    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        if (calls.fetch_add(1, std::memory_order_relaxed) >= fail_at) return false;
        return real->copy_blob(layer, expert, dst);
    }
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override {
        if (calls.fetch_add(1, std::memory_order_relaxed) >= fail_at) return false;
        return real->copy_blobs(layers, experts, dst, n);
    }
};

struct PrefillState {
    int64_t T_cap = 0, K = 0, N = 0, FF = 0;
    // device workspace (allocated once per T_cap)
    void* xq = nullptr;
    void* hq = nullptr;
    int32_t* bounds = nullptr;
    int32_t* rel = nullptr;
    int32_t* ident = nullptr;
    int32_t* src = nullptr;
    int32_t* entry_of_tok = nullptr;
    float* gu = nullptr;
    float* h = nullptr;
    float* dm = nullptr;
    uint8_t* grp_gu = nullptr;
    uint8_t* grp_d = nullptr;
    size_t gu_group = 0, d_group = 0;
    size_t max_gu_group = 0, max_d_group = 0;
    int64_t rel_cap = 0;
    // the device ring: one block, the slots are views into it
    uint8_t* ring_block = nullptr;
    // the prompt path's loan (expert_native_loan_plan / expert_native_loan_return)
    bool borrowed = false;       ///< every device buffer lives inside the cache's lent tail
    bool loan_started = false;   ///< a loan was planned for this state
    bool loan_returned = false;  ///< the loan was given back (the workspace must not be reused)
    uint64_t loan_bytes = 0;     ///< the borrowed region's bytes (variable-slot aware)
    uint64_t owned_bytes = 0;    ///< the device bytes this state allocated itself
    std::string loan_reason;     ///< why owned buffers were used (empty when borrowed)
    // host staging (pinned): ids, per-entry maps and the cold ring
    uint8_t* ids_raw = nullptr;      ///< the raw strided id columns as copied back
    int64_t ids_raw_bytes = 0;
    float* weights_h = nullptr;
    float* weights_dev = nullptr;
    uint8_t* weights_raw = nullptr;
    int64_t weights_raw_bytes = 0;
    int32_t* src_h = nullptr;
    int32_t* entry_h = nullptr;
    uint8_t** ring_d = nullptr;
    int ring = 0;
    size_t blob_cap = 0;
    cudaStream_t copy = nullptr;
    cudaEvent_t* copied = nullptr;
    cudaEvent_t* used = nullptr;
    std::unique_ptr<strata::prefill::mmq::Context> ctx;
    // the original cold-expert producer (one per allocation epoch): it owns the host ring that replaces the
    // old ring_h; the launching thread (this one) still issues every H2D and owns the device ring
    std::unique_ptr<strata::prefill::ColdStager> stager;
    int stager_threads = 1;  ///< the producer count chosen at init (1 default; 2/3 only by the exact env)
    std::vector<strata::prefill::ColdStager::Job> jobs;   ///< reused per layer (bounded by n_expert)
    std::vector<int32_t> job_of;                          ///< order index -> cold job index (-1 resident)
    StagerFaultSource fault;                              ///< the controlled negative's source (default unused)
    // bounded cold-page hint diagnostics (hint-on only; untouched on the default path)
    int64_t ch_calls = 0, ch_success = 0, ch_pairs = 0;
    uint64_t ch_bytes = 0, ch_wall_us = 0;
    int stg_chk_n = 0;
    bool stg_chk_failed = false;
    StagerChkPair stg_chk_pairs[kStagerChkMax];
    // route-boundary probes (diagnostic; alloc_for clears; host memory only allocated when a capture happens)
    int64_t probe_call = 0;                  ///< prompt-call ordinal (same order as the route fp calls)
    int probe_seen16 = 0, probe_seen17 = 0;  ///< T==128 occurrences of layers 16/17
    std::vector<uint8_t> probe_buf;          ///< three bounded host arrays (A at 0, C at cap, B at 2*cap)
    int ring_next = 0;
    int64_t resident = 0, streamed = 0, product_calls = 0;
    uint64_t stream_bytes = 0;
    uint64_t workspace_bytes = 0, ring_bytes = 0;
    // routing fingerprint (diagnostic; alloc_for clears it, so it is per request)
    int64_t fp_calls = 0;
    uint64_t fp_total = kFnvSeed;
    bool fp_loan_done = false;
    uint64_t fp_loan_hash = 0;
    int64_t fp_loan_slots = 0, fp_loan_first = 0;
    uint64_t fp_loan_bytes = 0;
    std::vector<RouteFpRec> fp_recs;
    // optional host phase accumulators (diagnostic; alloc_for clears them; host wall at existing boundaries)
    int64_t ph_calls = 0;
    double ph_d2h_ms = 0, ph_group_ms = 0, ph_ringwait_ms = 0, ph_copyhost_ms = 0, ph_enqueue_ms = 0,
           ph_combine_ms = 0;
    // bounded per-call GPU event pairs (diagnostic; STRATA_QWEN35_GPU_EVENTS=1; off leaves all of it unused)
    // Timing events (elapsed-able, created with cudaEventDefault); the stream-sync events copied/used stay
    // cudaEventDisableTiming.  Every failure is counted and reported; a missing measurement is never a zero.
    cudaEvent_t evt[256] = {};
    bool evt_ready = false;
    int evt_pairs = 0;
    uint8_t evt_tag[128] = {};
    double ev_phase_ms[8] = {};
    int64_t ev_phase_ok[8] = {};       // pairs whose elapsed was read successfully
    int64_t ev_phase_fail[8] = {};     // pairs whose elapsed failed (NOT counted as success)
    int64_t ev_phase_expected[8] = {}; // phase pairs entered (record attempted; includes create/cap failures)
    int64_t ev_launch[8] = {};         // kernel launches enqueued per phase (gather fallback = ngx launches)
    int64_t ev_calls = 0;              // calls whose full pair set collected with no elapsed failure
    int64_t ev_calls_failed = 0;       // calls with an elapsed failure or a tail sync failure
    int64_t ev_span_fail = 0;          // per-call span elapsed failures
    int64_t ev_truncated = 0;          // pairs not recorded: the 128-pair cap was hit
    int64_t ev_record_failed = 0;      // cudaEventRecord failures (start or end)
    int64_t ev_create_failed = 0;      // event handles that failed to create (whole set disabled)
    int64_t ev_dropped = 0;            // pending pairs dropped at a workspace rebuild or an unconfirmable tail
    bool ev_pending_unknown = false;   // the tail collection could not be confirmed (reported as unknown)
    bool ev_disabled = false;          // an elapsed/tail failure stopped the diagnostic for this request
    bool ev_error_reported = false;    // the stop reason was printed once
    // per-layer routing-row histogram (diagnostic; STRATA_QWEN35_ROWHIST=1; host only, summarized at close)
    std::vector<int64_t> rh_distinct, rh_maxcnt, rh_rows, rh_groups;
    int64_t rh_buckets[12] = {};
    int64_t rh_layers = 0;
    // one-shot single-layer capture (diagnostic; STRATA_QWEN35_LAYER_CAPTURE=<dir>)
    bool cap_done = false;
    bool cap_this_call = false;
    // optional group-gather counters (diagnostic; alloc_for clears them) and the byte-check pair registry
    int64_t gg_fast = 0, gg_fallback = 0, gg_kernels = 0, gg_used_recs = 0;
    int chk_n = 0;
    int chk_skipped = 0;
    bool chk_failed = false;
    GatherChkPair chk_pairs[kGatherChkMax];
    std::vector<int32_t> cnt, off, order;
};

PrefillState* state(ExpertDispatch& d) { return (PrefillState*) d.prefill_state; }

// Bounded GPU stream-event pairs (STRATA_QWEN35_GPU_EVENTS=1, default off): at most 128 pairs per call
// (xquant; per-group gather / GU product / activation / DOWN product; combine), recorded on the work stream
// and collected on the NEXT call right after its existing D2H sync (same stream, so the records are complete
// by then).  The tail collects after ONE last-event wait, never a per-kernel sync.  An elapsed span is a
// STREAM span: it includes launch gaps and any stream waits enqueued between the pair - not SM busy time.
// The timing events are created with cudaEventDefault (cudaEventElapsedTime rejects DisableTiming events);
// the copied/used ring events stay DisableTiming because they are only ever waited on.
enum EvPhase { EV_XQ = 0, EV_GATHER, EV_GU, EV_ACT, EV_DN, EV_COMBINE, EV_SPAN, EV_PHASES };
constexpr int kEvMaxPairs = 128;
/// Starts one phase pair and tries to record its start event.  The expected count rises for every entered
/// phase (so an unrecordable pair is an explicit missing measurement, not a silent zero); the cap overflow
/// is counted as truncated.  Returns true when the caller must record the end event.
bool ev_pair_begin(PrefillState& s, cudaStream_t cs, int tag) {
    const int t = tag < EV_PHASES ? tag : 0;
    ++s.ev_phase_expected[t];
    if (!s.evt_ready || s.ev_disabled) return false;
    if (s.evt_pairs >= kEvMaxPairs) { ++s.ev_truncated; return false; }
    s.evt_tag[s.evt_pairs] = (uint8_t) tag;
    if (cudaEventRecord(s.evt[2 * s.evt_pairs], cs) != cudaSuccess) {
        ++s.ev_record_failed;
        return false;
    }
    return true;
}
bool ev_pair_end(PrefillState& s, cudaStream_t cs) {
    if (!s.evt_ready || s.ev_disabled || s.evt_pairs >= kEvMaxPairs) return false;
    if (cudaEventRecord(s.evt[2 * s.evt_pairs + 1], cs) != cudaSuccess) {
        ++s.ev_record_failed;
        return false;
    }
    ++s.evt_pairs;
    return true;
}
/// The caller must have synchronized the stream or the last recorded event.  Every pair's elapsed is read and
/// classified: read-ok into its phase total, read-fail into ev_phase_fail (NOT a zero, NOT a success).  A call
/// counts as collected only when the whole pair set plus the span elapsed cleanly; otherwise it is counted
/// failed and the summary reports UNKNOWN.  On any elapsed failure the diagnostic STOPS for this request and
/// prints the failure once; the error is NOT cleared - cudaGetLastError() would return this host thread's last
/// runtime error, which may be an earlier asynchronous kernel error, so consuming it here could hide a real
/// failure.  A stop leaves every unread pair counted as dropped, never as a completed measurement.
void ev_collect(PrefillState& s) {
    if (s.evt_pairs <= 0) return;
    const int pending = s.evt_pairs;
    int failed = 0;
    int p = 0;
    for (; p < pending; ++p) {
        float ms = 0.0f;
        const cudaError_t e = cudaEventElapsedTime(&ms, s.evt[2 * p], s.evt[2 * p + 1]);
        const int t = s.evt_tag[p] < EV_PHASES ? s.evt_tag[p] : 0;
        if (e != cudaSuccess) {
            ++s.ev_phase_fail[t];
            ++failed;
            if (!s.ev_error_reported) {
                s.ev_error_reported = true;
                std::fprintf(stderr, "strata qwen35: gpu events: pair %d (phase %d) elapsed failed: %s; the event "
                                     "diagnostic stops for this request and reports UNKNOWN (the error is NOT "
                                     "cleared - it may be a real earlier async error)\n",
                             p, t, cudaGetErrorString(e));
            }
            s.ev_disabled = true;
            break;
        }
        s.ev_phase_ms[t] += ms;
        ++s.ev_phase_ok[t];
    }
    if (failed != 0) {
        // The failing pair is counted failed; the pairs after it were recorded but never read: dropped.
        s.ev_dropped += (pending - p - 1 > 0 ? pending - p - 1 : 0);
    }
    if (failed == 0) {
        float span = 0.0f;
        if (cudaEventElapsedTime(&span, s.evt[0], s.evt[2 * s.evt_pairs - 1]) == cudaSuccess) {
            s.ev_phase_ms[EV_SPAN] += span;
        } else {
            ++s.ev_span_fail;
            ++failed;
            s.ev_disabled = true;
        }
    }
    if (failed == 0) ++s.ev_calls; else ++s.ev_calls_failed;
    s.evt_pairs = 0;
}
/// 12 fixed buckets for routed rows per expert: 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64-127 | 128-255 |
/// 256-511 | 512-1023 | 1024-2047 | 2048+.
int row_bucket(int64_t c) {
    if (c <= 1) return 0;
    if (c <= 3) return 1;
    if (c <= 7) return 2;
    if (c <= 15) return 3;
    if (c <= 31) return 4;
    if (c <= 63) return 5;
    if (c <= 127) return 6;
    if (c <= 255) return 7;
    if (c <= 511) return 8;
    if (c <= 1023) return 9;
    if (c <= 2047) return 10;
    return 11;
}

void free_all(PrefillState& s) {
    const bool borrowed = s.borrowed;
    void* ptrs[] = {s.xq, s.hq, s.gu, s.h, s.dm, s.grp_gu, s.grp_d};
    for (int i = 0; i < 7; ++i)
        if (ptrs[i] != nullptr && !borrowed) cudaFree(ptrs[i]);
    s.xq = s.hq = nullptr;
    s.gu = s.h = s.dm = nullptr;
    s.grp_gu = s.grp_d = nullptr;
    int32_t* ip[] = {s.bounds, s.rel, s.ident, s.src, s.entry_of_tok};
    for (int i = 0; i < 5; ++i)
        if (ip[i] != nullptr && !borrowed) cudaFree(ip[i]);
    s.bounds = s.rel = s.ident = s.src = s.entry_of_tok = nullptr;
    if (s.ids_raw) { cudaFreeHost(s.ids_raw); s.ids_raw = nullptr; s.ids_raw_bytes = 0; }
    if (s.weights_h) { cudaFreeHost(s.weights_h); s.weights_h = nullptr; }
    if (s.weights_raw) { cudaFreeHost(s.weights_raw); s.weights_raw = nullptr; s.weights_raw_bytes = 0; }
    if (s.weights_dev) { if (!borrowed) cudaFree(s.weights_dev); s.weights_dev = nullptr; }
    if (s.src_h) { cudaFreeHost(s.src_h); s.src_h = nullptr; }
    if (s.entry_h) { cudaFreeHost(s.entry_h); s.entry_h = nullptr; }
    if (s.ring_d) {
        delete[] s.ring_d;   // views into ring_block (never owned per slot)
        s.ring_d = nullptr;
    }
    if (s.ring_block) {
        if (!borrowed) cudaFree(s.ring_block);
        s.ring_block = nullptr;
    }
    if (s.copied) {
        for (int i = 0; i < s.ring; ++i) if (s.copied[i]) cudaEventDestroy(s.copied[i]);
        delete[] s.copied;
        s.copied = nullptr;
    }
    if (s.used) {
        for (int i = 0; i < s.ring; ++i) if (s.used[i]) cudaEventDestroy(s.used[i]);
        delete[] s.used;
        s.used = nullptr;
    }
    // The bounded GPU-event diagnostic's event objects (created only when the gate was on for this epoch); a
    // pending pair from a failed/early call is explicitly counted as dropped here - it is never queried after
    // destruction and it is never reported as a completed measurement.
    if (s.evt_pairs > 0) {
        s.ev_dropped += s.evt_pairs;
        s.evt_pairs = 0;
    }
    for (int i = 0; i < 256; ++i) {
        if (s.evt[i] != nullptr) { cudaEventDestroy(s.evt[i]); s.evt[i] = nullptr; }
    }
    s.evt_ready = false;
    if (s.copy) { cudaStreamDestroy(s.copy); s.copy = nullptr; }
    // The producer's pinned ring dies with its instance; every caller of free_all has drained the copy stream
    // first (alloc_for syncs it; close syncs it), so no in-flight DMA can still read these host buffers.
    s.stager.reset();
    s.jobs.clear();
    s.job_of.clear();
    s.ring = 0;
    s.T_cap = 0;
    s.workspace_bytes = s.ring_bytes = 0;
}

bool alloc_for(ExpertDispatch& d, PrefillState& s, const strata::kernels::cpu::NativeFmt& f, int64_t T) {
    const int64_t N = f.n_embd, FF = f.n_ff, K = d.native_k > 0 ? d.native_k : s.K;
    const int64_t n = T * K;
    if (s.loan_started && !s.loan_returned) {
        // The current buffers sit inside a live loan: give them (and their experts) back before re-sizing.
        // A failed return must STOP the re-size - the borrowed bytes stay the cache's until a retry (the
        // shutdown path tries once more), never silently re-planned under a live loan.
        std::string e2;
        if (!qwen_prefill_loan_return(d, e2)) {
            std::fprintf(stderr, "strata qwen35: the workspace re-size stopped: the loan was not returned: %s\n",
                         e2.c_str());
            return false;
        }
    }
    // The previous epoch's producer buffers are about to die: the copy stream must be CHECKED idle so no
    // in-flight H2D can still read the pinned ring (the loan-return above already synced the device when a
    // loan was live; the owned path has only this stream's work).  A failed sync gets ONE explicit best-effort
    // checked fallback; if that also fails the re-size stops with everything retained (close re-checks with
    // the same gate and applies the fatal rule if the drain stays unconfirmable).
    if (s.copy != nullptr) {
        const cudaError_t ce = cudaStreamSynchronize(s.copy);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: the prompt copy stream did not drain before a workspace re-size: %s\n",
                         cudaGetErrorString(ce));
            const cudaError_t fb = cudaDeviceSynchronize();
            if (fb != cudaSuccess) {
                std::fprintf(stderr, "strata qwen35: the fallback device sync failed too (%s); the re-size stops and "
                                     "nothing is released\n",
                             cudaGetErrorString(fb));
                return false;
            }
            std::fprintf(stderr, "strata qwen35: fallback checked device sync succeeded; releasing the old epoch\n");
        }
    }
    free_all(s);
    // A new workspace is a new request: the fingerprint's bounded records reset here.
    s.fp_calls = 0;
    s.fp_total = kFnvSeed;
    s.fp_loan_done = false;
    s.fp_loan_hash = 0;
    s.fp_recs.clear();
    s.ph_calls = 0;
    s.ph_d2h_ms = s.ph_group_ms = s.ph_ringwait_ms = s.ph_copyhost_ms = s.ph_enqueue_ms = s.ph_combine_ms = 0;
    s.gg_fast = s.gg_fallback = s.gg_kernels = s.gg_used_recs = 0;
    s.chk_n = 0;
    s.chk_skipped = 0;
    s.chk_failed = false;
    // The GPU-event diagnostic resets per request; the events themselves are created below (gate on only).
    for (int i = 0; i < 8; ++i) {
        s.ev_phase_ms[i] = 0;
        s.ev_phase_ok[i] = s.ev_phase_fail[i] = s.ev_phase_expected[i] = s.ev_launch[i] = 0;
    }
    s.ev_calls = s.ev_calls_failed = s.ev_span_fail = s.ev_truncated = s.ev_record_failed = 0;
    s.ev_dropped = 0;
    s.ev_pending_unknown = false;
    s.ev_disabled = false;
    s.ev_error_reported = false;
    s.evt_pairs = 0;
    if (gpu_events_enabled()) {
        // Timing events: cudaEventDefault (cudaEventElapsedTime rejects DisableTiming events).  Only these 256
        // are timing events; the copied/used ring events below keep cudaEventDisableTiming (they are waited on).
        int failed = 0;
        for (int i = 0; i < 256; ++i)
            if (cudaEventCreateWithFlags(&s.evt[i], cudaEventDefault) != cudaSuccess) { ++failed; s.evt[i] = nullptr; }
        s.ev_create_failed = failed;
        s.evt_ready = failed == 0;
        if (failed != 0) {
            for (int i = 0; i < 256; ++i)
                if (s.evt[i] != nullptr) { cudaEventDestroy(s.evt[i]); s.evt[i] = nullptr; }
            std::fprintf(stderr, "strata qwen35: the GPU event diagnostic could not create its timing events "
                                 "(%d failed); disabled for this workspace (measurements will be reported as "
                                 "unknown, not as zero)\n",
                         failed);
        }
    } else {
        s.evt_ready = false;
    }
    // The routing-row histogram resets per request (host only).
    for (int64_t& b : s.rh_buckets) b = 0;
    s.rh_layers = 0;
    s.K = K; s.N = N; s.FF = FF; s.T_cap = T;
    s.gu_group = strata::prefill::mmq::matrix_bytes(f.gu_type, 2 * FF, N);
    s.d_group = strata::prefill::mmq::matrix_bytes(f.d_type, N, FF);
    // The group buffers are sized for the LARGEST layer format (layers mix IQ2_S/IQ3_XXS gate/up and
    // IQ3_S/IQ4_XS down); the strides above stay the current layer's.
    size_t max_gu = 0, max_d = 0;
    for (const auto& f2 : strata::kernels::cpu::expert_layout().fmt) {
        max_gu = std::max(max_gu, strata::prefill::mmq::matrix_bytes(f2.gu_type, 2 * f2.n_ff, f2.n_embd));
        max_d = std::max(max_d, strata::prefill::mmq::matrix_bytes(f2.d_type, f2.n_embd, f2.n_ff));
    }
    s.max_gu_group = max_gu;
    s.max_d_group = max_d;
    const size_t xq_bytes = strata::prefill::mmq::q8_bytes(n, N);
    const size_t hq_bytes = strata::prefill::mmq::q8_bytes(n, FF);
    const size_t gu_bytes = (size_t) n * (size_t) (2 * FF) * sizeof(float);
    const size_t h_bytes = (size_t) n * (size_t) FF * sizeof(float);
    const size_t dm_bytes = (size_t) n * (size_t) N * sizeof(float);
    s.rel_cap = (n / kGroup + 2) * (kGroup + 1);
    const auto& lay = strata::kernels::cpu::expert_layout();
    s.blob_cap = (size_t) lay.max_blob;
    s.ring = (int) std::min<int64_t>(kGroup, lay.n_expert);
    const size_t bounds_bytes = (size_t) (n + 1) * sizeof(int32_t);
    const size_t rel_bytes = (size_t) s.rel_cap * sizeof(int32_t);
    const size_t idx_bytes = (size_t) n * sizeof(int32_t);
    const size_t wdev_bytes = (size_t) n * sizeof(float);
    const size_t grp_gu_bytes = (size_t) kGroup * max_gu + kTail;
    const size_t grp_d_bytes = (size_t) kGroup * max_d + kTail;
    const size_t ring_dev_bytes = (size_t) s.ring * s.blob_cap;
    s.workspace_bytes = xq_bytes + hq_bytes + bounds_bytes + rel_bytes + 3 * idx_bytes + gu_bytes + h_bytes +
                        dm_bytes + wdev_bytes + grp_gu_bytes + grp_d_bytes;
    // host+device accounting stays 2x (the old ring_h and the producer's ring are the same 16 x blob_cap
    // pinned bytes); only the DEVICE half is lent/priced - the host ring is never part of the device loan.
    s.ring_bytes = 2 * (uint64_t) s.ring * s.blob_cap;
    // THE LOAN (same contract as generate.cpp's lent/host_res): with a cache, the tail slots are borrowed for
    // the workspace + device ring and their residents become misses until the return refills them; the region
    // is priced by the cache's REAL slot offsets, so a variable-size cache is priced by what it really holds.
    // Cache-off or a tail that cannot cover the block falls back to this state's own bounded buffers, with
    // the reason kept for the report.
    uint64_t borrowed_region = 0;
    s.loan_reason.clear();
    void* base = strata::core::expert_native_loan_plan(d, (uint64_t) s.workspace_bytes + ring_dev_bytes + 65536,
                                                       borrowed_region, s.loan_reason);
    s.borrowed = base != nullptr;
    s.loan_started = s.borrowed;
    s.loan_returned = false;
    s.loan_bytes = s.borrowed ? borrowed_region : 0;
    s.owned_bytes = 0;
    uint8_t* cursor = (uint8_t*) base;
    uint8_t* bend = s.borrowed ? cursor + borrowed_region : nullptr;
    auto take = [&](size_t bytes) -> void* {
        const uintptr_t at = ((uintptr_t) cursor + 255) & ~(uintptr_t) 255;
        if (bend == nullptr || at + bytes > (uintptr_t) bend) return nullptr;
        cursor = (uint8_t*) (at + bytes);
        return (void*) at;
    };
    bool ok = true;
    auto dc = [&](void** p, size_t bytes) {
        if (s.borrowed) { *p = take(bytes); ok = ok && *p != nullptr; }
        else { ok = ok && cudaMalloc(p, bytes) == cudaSuccess; }
    };
    dc(&s.xq, xq_bytes);
    dc(&s.hq, hq_bytes);
    dc((void**) &s.bounds, bounds_bytes);
    dc((void**) &s.rel, rel_bytes);
    dc((void**) &s.ident, idx_bytes);
    dc((void**) &s.src, idx_bytes);
    dc((void**) &s.entry_of_tok, idx_bytes);
    dc((void**) &s.gu, gu_bytes);
    dc((void**) &s.h, h_bytes);
    dc((void**) &s.dm, dm_bytes);
    dc((void**) &s.grp_gu, grp_gu_bytes);
    dc((void**) &s.grp_d, grp_d_bytes);
    dc((void**) &s.weights_dev, (size_t) n * sizeof(float));
    s.ring_d = new uint8_t*[s.ring]();
    s.copied = new cudaEvent_t[s.ring]();
    s.used = new cudaEvent_t[s.ring]();
    if (s.borrowed) {
        s.ring_block = (uint8_t*) take(ring_dev_bytes);
        ok = ok && s.ring_block != nullptr;
    } else {
        ok = ok && cudaMalloc((void**) &s.ring_block, ring_dev_bytes) == cudaSuccess;
    }
    for (int i = 0; i < s.ring; ++i)
        s.ring_d[i] = s.ring_block != nullptr ? s.ring_block + (size_t) i * s.blob_cap : nullptr;
    ok = ok && cudaHostAlloc((void**) &s.ids_raw, (size_t) n * sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess;
    // The pointer and its capacity bookkeeping move together: a stale (larger) `ids_raw_bytes` over a fresh
    // smaller buffer is exactly how a rebuild would write past the host allocation in the strided D2H below.
    s.ids_raw_bytes = s.ids_raw != nullptr ? (int64_t) ((size_t) n * sizeof(int32_t)) : 0;
    ok = ok && cudaHostAlloc((void**) &s.src_h, (size_t) n * sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &s.weights_h, (size_t) n * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &s.entry_h, (size_t) n * sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess;
    for (int i = 0; i < s.ring && ok; ++i) {
        ok = cudaEventCreateWithFlags(&s.copied[i], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&s.used[i], cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaEventRecord(s.used[i], (cudaStream_t) 0) == cudaSuccess;
    }
    ok = ok && cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking) == cudaSuccess;
    // The ORIGINAL ColdStager producer (no Qwen-specific stager): one pinned 16-buffer ring, one worker that
    // only does host reads, explicit opt-in policy.  The launching thread still issues every H2D below.
    s.job_of.clear();
    s.stg_chk_n = 0;
    s.stg_chk_failed = false;
    s.probe_call = 0;
    s.probe_seen16 = 0;
    s.probe_seen17 = 0;
    s.probe_buf.clear();
    s.stager = std::make_unique<strata::prefill::ColdStager>();
    {
        strata::prefill::ColdStagerPolicy pol;
        pol.ring = s.ring;          // 16: the same host ring the old ring_h held (2x accounting unchanged)
        pol.sleep = 1;              // sleeping issued waits + BlockingSync dma_done events
        pol.strict_pinned = true;   // pinned or fail (no pageable fallback on the GPU prompt path)
        pol.propagate_error = true; // latch failures, wait() -> nullptr, no abort
        pol.perf_counters = stager_stats_enabled();   // opt-in host-wall scopes, default off (no clocks/atomics)
        // Write-combined payload ring: exact env only (default off).  The stager refuses unsupported platforms,
        // verifies the real cudaHostAllocWriteCombined flags and never falls back to a cached/pageable ring;
        // the WC + STAGER_CHECK combination is refused before the stager is created.
        pol.write_combined = stager_wc_requested();
        // Exact-value opt-in: "2" or "3" select that many producers; unset or any other value keeps the frozen
        // single-producer default (no clamp grid, no default change; 4/5 are not accepted).
        const char* stg_env = std::getenv("STRATA_QWEN35_STAGER_THREADS");
        s.stager_threads = (stg_env != nullptr && std::strcmp(stg_env, "2") == 0) ? 2
                         : (stg_env != nullptr && std::strcmp(stg_env, "3") == 0) ? 3 : 1;
        ok = ok && s.stager->init(s.blob_cap, s.stager_threads, pol);
    }
    {
        std::vector<int32_t> ident((size_t) n);
        for (int64_t i = 0; i < n; ++i) ident[(size_t) i] = (int32_t) i;
        ok = ok && cudaMemcpy(s.ident, ident.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        if (s.borrowed) {
            // The tail (and its experts) go back to the cache; the borrowed bytes are never cudaFree'd.  No
            // producer work was started and no DMA was enqueued on the fresh stream, so no drain is needed
            // here; a failed return leaves the loan live for the close path's checked/fatal gate.
            std::string e2;
            if (!qwen_prefill_loan_return(d, e2))
                std::fprintf(stderr, "strata qwen35: the half-built prompt workspace could not return its loan: %s\n",
                             e2.c_str());
        }
        free_all(s);
        return false;
    }
    if (!s.borrowed) s.owned_bytes = s.workspace_bytes + ring_dev_bytes;
    d.native_stats.prefill_workspace_bytes = s.workspace_bytes;
    d.native_stats.prefill_ring_bytes = s.ring_bytes;
    d.native_stats.prefill_owned_bytes = s.owned_bytes;
    d.native_stats.prefill_loan_bytes = s.borrowed ? s.loan_bytes : 0;
    d.native_stats.prefill_loan_slots = s.borrowed ? d.native_loan_slots : 0;
    d.native_stats.prefill_loan_why = s.loan_reason;
    if (s.ctx == nullptr) s.ctx = std::make_unique<strata::prefill::mmq::Context>();
    return true;
}

__global__ void combine_scatter_kernel(const float* __restrict__ dm, const float* __restrict__ w,
                                       const int32_t* __restrict__ entry_of_tok, float* __restrict__ out,
                                       int64_t N, int K) {
    const int64_t t = blockIdx.x;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < K; ++r) {
            const int32_t e = entry_of_tok[t * K + r];
            acc += w[t * K + r] * dm[(int64_t) e * N + j];
        }
        out[t * N + j] = acc;
    }
}

}  // namespace

bool qwen_prefill_mmq_available() { return strata::prefill::mmq::built(); }

bool qwen_prefill_experts(ExpertDispatch& d, int64_t layer, int64_t T, const void* x_dev, int64_t x_nb1,
                          const void* ids_dev, int64_t ids_nb1, const void* weights_dev, int64_t w_nb1,
                          int64_t w_nb2, void* out_dev, void* stream, std::string& err) {
    using namespace strata::kernels::cpu;
    namespace mmq = strata::prefill::mmq;
    if (d.failed) return false;
    if (T <= 0) return true;   // a cropped prompt ubatch (no output token at layer 39): nothing to compute
    if (!mmq::built()) { err = "the prompt expert path needs the MMQ build"; return false; }
    const ExpertLayout& lay = expert_layout();
    if (!lay.native || layer < 0 || layer >= (int64_t) lay.fmt.size()) {
        err = "the prompt expert path needs a native pack";
        return false;
    }
    const NativeFmt& f = lay.fmt[(size_t) layer];
    if (f.n_embd != d.n_embd || f.n_ff != d.n_ff) {
        err = "the prompt expert path's geometry does not match the pack";
        return false;
    }
    if (d.native_k <= 0) { err = "the prompt expert path has no configured k"; return false; }
    const int64_t k = d.native_k;
    if (T < 1 || T > 4096) { err = "the prompt expert path's token count is out of range"; return false; }
    const int64_t n = T * k;
    PrefillState* s = state(d);
    if (s == nullptr) {
        s = new PrefillState();
        d.prefill_state = s;
    }
    if (stager_wc_requested() && stager_check_enabled()) {
        // The stager byte check memcmp's the staging payload on the CPU; write-combined memory is a host-write/
        // device-read buffer, so the combination is refused explicitly (a cached fallback would not validate WC).
        err = "STRATA_QWEN35_STAGER_WC and STRATA_QWEN35_STAGER_CHECK cannot be combined (the check reads the "
              "staging payload on the CPU)";
        d.failed = true;
        return false;
    }
    if (s->T_cap < T || s->N != f.n_embd || s->FF != f.n_ff || s->K != k || s->loan_returned) {
        if (!alloc_for(d, *s, f, T)) { err = "the prompt expert workspace could not be allocated"; return false; }
    }
    // Per-layer strides (the buffers are sized for the largest format).
    s->gu_group = strata::prefill::mmq::matrix_bytes(f.gu_type, 2 * f.n_ff, f.n_embd);
    s->d_group = strata::prefill::mmq::matrix_bytes(f.d_type, f.n_embd, f.n_ff);
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "qwen prefill trace: layer %lld T %lld x_nb1 %lld enter (ring %d, blob %zu)\n", (long long) layer, (long long) T, (long long) x_nb1, s->ring, s->blob_cap);
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t cs = (cudaStream_t) stream;
    const bool ph = prefill_phases_enabled();
    const bool gg_stats = gather_stats_enabled();
    const bool gg_chk = gather_check_enabled();
    const bool stg_chk = stager_check_enabled();
    // Route-boundary probes (default off): A/C are the layer's expert INPUT (attn_post_norm: T rows of H f32
    // with stride x_nb1) at the SECOND T==128 prompt ubatch; B is layer 16's combine OUTPUT read after the
    // combine on the same work stream.  Only flags/counters here; the copies ride the existing D2H sync (A/C)
    // and one bounded post-combine sync (B).
    const bool probe = route_probe_enabled();
    bool probe_a = false, probe_c = false, probe_b = false;
    if (probe) {
        if (T == 128 && layer == 16) probe_a = (++s->probe_seen16 == 2);
        else if (T == 128 && layer == 17) probe_c = (++s->probe_seen17 == 2);
        probe_b = probe_a;
        if (probe_a || probe_c) {
            const size_t cap = (size_t) T * (size_t) f.n_embd * sizeof(float);
            if (s->probe_buf.size() < 3 * cap) s->probe_buf.assign(3 * cap, 0);
        }
    }
    const int64_t probe_call = probe ? s->probe_call++ : -1;
    // Router ids: one small D2H per prompt layer (the grouping needs them on the host).  The ids tensor is
    // llama's strided argsort view, so the raw columns come back and the host reads them at `ids_nb1` stride.
    const size_t ids_raw_bytes = (size_t) (T - 1) * (size_t) ids_nb1 + (size_t) k * sizeof(int32_t);
    if (s->ids_raw_bytes < ids_raw_bytes) {
        if (s->ids_raw) cudaFreeHost(s->ids_raw);
        if (cudaHostAlloc((void**) &s->ids_raw, ids_raw_bytes, cudaHostAllocDefault) != cudaSuccess) {
            s->ids_raw = nullptr;
            s->ids_raw_bytes = 0;
            err = "the prompt expert path's id staging could not be allocated";
            return false;
        }
        s->ids_raw_bytes = ids_raw_bytes;
    }
    const size_t weights_raw_bytes = (size_t) (T - 1) * (size_t) w_nb2 + (size_t) (k - 1) * (size_t) w_nb1 + sizeof(float);
    if (s->weights_raw_bytes < (int64_t) weights_raw_bytes) {
        if (s->weights_raw) cudaFreeHost(s->weights_raw);
        if (cudaHostAlloc((void**) &s->weights_raw, weights_raw_bytes, cudaHostAllocDefault) != cudaSuccess) {
            s->weights_raw = nullptr;
            s->weights_raw_bytes = 0;
            err = "the prompt expert path's weight staging could not be allocated";
            return false;
        }
        s->weights_raw_bytes = (int64_t) weights_raw_bytes;
    }
    // ONE work-stream sync covers BOTH original strided D2H copies (ids first, then weights): the ids wait that
    // used to sit between them was redundant - both copies ride `cs`, whose `pre` ordering (recorded by the
    // bridge) already gates the upstream graph.  Strides, pinned capacities and failure handling are the
    // originals; no cudaDeviceSynchronize and no other stream's data is read early.
    {
        PhaseScope ph_d2h(ph, &s->ph_d2h_ms);
        if (cudaMemcpyAsync(s->ids_raw, ids_dev, ids_raw_bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            err = "the prompt expert path's router ids did not transfer";
            return false;
        }
        if (cudaMemcpyAsync(s->weights_raw, weights_dev, weights_raw_bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            err = "the prompt expert path's router weights did not transfer";
            return false;
        }
        // Route probe A/C (default off): the expert input is read on the work stream at the same sync as the
        // router ids/weights - valid because `pre` orders the graph prefix before both.  Only the H valid f32 of
        // each row (stride x_nb1) are copied, never row padding.
        if (probe && (probe_a || probe_c)) {
            const size_t cap = (size_t) T * (size_t) f.n_embd * sizeof(float);
            const size_t row = (size_t) f.n_embd * sizeof(float);
            uint8_t* dst = s->probe_buf.data() + (probe_c ? cap : 0);
            if (cudaMemcpy2DAsync(dst, row, x_dev, (size_t) x_nb1, row, (size_t) T,
                                  cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                err = "the route probe's input copy failed";
                d.failed = true;
                return false;
            }
        }
        if (cudaStreamSynchronize(cs) != cudaSuccess) {
            err = "the prompt expert path's router ids/weights did not complete";
            return false;
        }
    }
    if (ph) ++s->ph_calls;
    // GPU-event diagnostic (default off): the D2H sync above completed every record of the PREVIOUS call on
    // this same stream, so this is the collect-and-reuse point.  Off: no call, no state.
    const bool gpu_ev = gpu_events_enabled();
    if (gpu_ev && s->evt_pairs > 0) ev_collect(*s);
    // Row histogram (default off): host counts only, accumulated per layer and printed once at close.
    const bool rowhist = rowhist_enabled();
    // One-shot layer capture (default off): first T==2048 call of layer 16 only.
    const char* cap_dir = layer_capture_dir();
    s->cap_this_call = cap_dir != nullptr && (int64_t) layer == 16 && T == 2048 && !s->cap_done;
    auto id_at = [&](int64_t t, int64_t r) {
        int32_t v;
        std::memcpy(&v, s->ids_raw + (size_t) t * (size_t) ids_nb1 + (size_t) r * sizeof(int32_t), sizeof(v));
        return v;
    };
    const std::chrono::steady_clock::time_point ph_group0 =
        ph ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (int64_t t = 0; t < T; ++t)
        for (int64_t r = 0; r < k; ++r)
            std::memcpy(&s->weights_h[(size_t) (t * k + r)],
                        s->weights_raw + (size_t) t * (size_t) w_nb2 + (size_t) r * (size_t) w_nb1, sizeof(float));
    if (cudaMemcpyAsync(s->weights_dev, s->weights_h, (size_t) n * sizeof(float), cudaMemcpyHostToDevice, cs) !=
        cudaSuccess) {
        err = "the prompt expert path's router weights did not upload";
        return false;
    }
    if (trace) std::fprintf(stderr, "qwen prefill trace: ids back\n");
    // Group the entries by expert (ascending), building the expert-sorted row maps.
    s->cnt.assign((size_t) d.n_expert, 0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t r = 0; r < k; ++r) {
            const int32_t e = id_at(t, r);
            if (e < 0 || e >= d.n_expert) { err = "a routed expert id is out of range"; d.failed = true; return false; }
            ++s->cnt[(size_t) e];
        }
    s->off.assign((size_t) d.n_expert, 0);
    int64_t at = 0;
    for (int64_t e = 0; e < d.n_expert; ++e) { s->off[(size_t) e] = (int32_t) at; at += s->cnt[(size_t) e]; }
    s->order.clear();
    for (int64_t e = 0; e < d.n_expert; ++e) if (s->cnt[(size_t) e] > 0) s->order.push_back((int32_t) e);
    // Opt-in count-descending row layout (exact STRATA_QWEN35_EXPERT_ORDER=count_desc; default untouched): the
    // non-empty experts are ordered by descending routed row count (ties by ascending expert ID), and the row
    // maps are rebuilt from that order - off is recomputed as the new per-expert starts (expert-indexed, no
    // tail entry of its own) and the cursor loop below fills src_h/entry_h in the same (t, r) scan order, so
    // every expert's rows keep their stable order and the combine scatter mapping is unchanged.  Only this
    // host layout changes: cnt/ids/weights/slots, group16, MMQ and the math are untouched.  Zero-count experts
    // keep the ascending pass's off values, which no consumer reads (cursor/bh only touch non-empty experts).
    if (expert_order_count_desc() && !s->order.empty()) {
        std::sort(s->order.begin(), s->order.end(), [&](int32_t a, int32_t b) {
            const int64_t ca = s->cnt[(size_t) a], cb = s->cnt[(size_t) b];
            return ca != cb ? ca > cb : a < b;
        });
        int64_t at2 = 0;
        for (int32_t e : s->order) { s->off[(size_t) e] = (int32_t) at2; at2 += s->cnt[(size_t) e]; }
    }
    {
        if (trace) {
        std::fprintf(stderr, "qwen prefill trace: ids_nb1 %lld w_nb1 %lld w_nb2 %lld; ids:", (long long) ids_nb1,
                     (long long) w_nb1, (long long) w_nb2);
        for (int64_t t = 0; t < T && t < 4; ++t)
            for (int64_t r = 0; r < k; ++r) std::fprintf(stderr, " %d", id_at(t, r));
        std::fprintf(stderr, "\n");
    }
    std::vector<int32_t> cursor = s->off;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t r = 0; r < k; ++r) {
                const int32_t e = id_at(t, r);
                const int32_t row = cursor[(size_t) e]++;
                s->src_h[(size_t) row] = (int32_t) t;
                s->entry_h[(size_t) (t * k + r)] = row;
            }
    }
    if (rowhist && !s->order.empty()) {
        if ((int64_t) s->rh_distinct.size() != (int64_t) lay.fmt.size()) {
            s->rh_distinct.assign(lay.fmt.size(), 0);
            s->rh_maxcnt.assign(lay.fmt.size(), 0);
            s->rh_rows.assign(lay.fmt.size(), 0);
            s->rh_groups.assign(lay.fmt.size(), 0);
        }
        int64_t mx = 0;
        for (int32_t e : s->order) {
            const int64_t c = s->cnt[(size_t) e];
            if (c > mx) mx = c;
            ++s->rh_buckets[row_bucket(c)];
        }
        s->rh_distinct[(size_t) layer] = (int64_t) s->order.size();
        s->rh_maxcnt[(size_t) layer] = mx;
        s->rh_rows[(size_t) layer] = n;
        s->rh_groups[(size_t) layer] = ((int64_t) s->order.size() + kGroup - 1) / kGroup;
        ++s->rh_layers;
    }
    if (route_fp_enabled()) {
        RouteFpRec rec;
        rec.call = s->fp_calls++;
        rec.layer = layer;
        rec.T = T;
        rec.k = k;
        rec.distinct = (int64_t) s->order.size();
        rec.groups = (rec.distinct + kGroup - 1) / kGroup;   // the call's non-empty groups (2 product calls each)
        // The same host `slot_of` decision the group loop makes; no residency changes during the prompt call.
        for (int32_t e : s->order) {
            if (d.cache != nullptr && d.cache->slot_of(layer, e) >= 0) ++rec.resident; else ++rec.streamed;
        }
        uint64_t ih = kFnvSeed;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t r = 0; r < k; ++r) { const int32_t v = id_at(t, r); ih = fnv_bytes(ih, &v, sizeof(v)); }
        uint64_t wh = kFnvSeed;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t r = 0; r < k; ++r) {
                float w;
                std::memcpy(&w, s->weights_raw + (size_t) t * (size_t) w_nb2 + (size_t) r * (size_t) w_nb1,
                            sizeof(w));
                wh = fnv_bytes(wh, &w, sizeof(w));
            }
        rec.ids_hash = ih;
        rec.weights_hash = wh;
        if (s->fp_recs.size() < 4096) s->fp_recs.push_back(rec);   // bounded (p512: 156 calls)
        s->fp_total = fnv_bytes(s->fp_total, &rec, sizeof(rec));
        if (!s->fp_loan_done) {
            s->fp_loan_done = true;
            s->fp_loan_slots = d.native_loan_slots;
            s->fp_loan_first = d.native_loan_first;
            s->fp_loan_bytes = d.native_loan_bytes;
            uint64_t lh = kFnvSeed;
            for (const auto& p : d.native_loan_lent) {
                lh = fnv_bytes(lh, &p.first, sizeof(p.first));
                lh = fnv_bytes(lh, &p.second, sizeof(p.second));
            }
            s->fp_loan_hash = lh;
        }
    }
    if (ph) s->ph_group_ms += ms_since(ph_group0);
    if (cudaMemcpyAsync(s->src, s->src_h, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(s->entry_of_tok, s->entry_h, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
            cudaSuccess) {
        err = "the prompt expert path's row maps did not upload";
        return false;
    }
    // The activations as MMQ q8_1 rows in expert order (one launch).
    if (trace) std::fprintf(stderr, "qwen prefill trace: maps up, quantize start\n");
    const bool ev_xq = gpu_ev && ev_pair_begin(*s, cs, EV_XQ);
    if (gpu_ev) ++s->ev_launch[EV_XQ];
    mmq::quantize((const float*) x_dev, s->src, s->xq, f.gu_type, f.n_embd, (int64_t) (x_nb1 / (int64_t) sizeof(float)), n, cs);
    if (ev_xq) ev_pair_end(*s, cs);
    if (trace && cudaDeviceSynchronize() != cudaSuccess)
        std::fprintf(stderr, "qwen prefill trace: QUANTIZE FAULT\n");
    if (trace) std::fprintf(stderr, "qwen prefill trace: quantize done, experts %lld\n", (long long) s->order.size());
    const int64_t n_experts = (int64_t) s->order.size();
    // The layer's cold experts through the ORIGINAL producer: one bounded job per cold expert in the same
    // order the group loop walks them (resident experts are not staged).  The worker only reads host memory;
    // this thread keeps issuing every H2D/DMA below (single issuer).
    s->jobs.clear();
    s->job_of.assign((size_t) n_experts, -1);
    // Bounded cold-page hint collection (default off): fixed stack array, no hot-loop allocation; only the
    // transient (GPU-miss, file-read) experts are collected - resident and stable-complement experts are not.
    const bool ch = cold_hint_enabled();
    std::pair<int32_t, int32_t> ch_pairs[256];
    int64_t ch_n = 0;
    uint64_t ch_bytes = 0;
    for (int64_t j = 0; j < n_experts; ++j) {
        const int32_t e = s->order[(size_t) j];
        const int32_t slot = d.cache != nullptr ? d.cache->slot_of(layer, e) : (int32_t) -1;
        if (slot >= 0) continue;
        s->job_of[(size_t) j] = (int32_t) s->jobs.size();
        if (d.src->transient(layer, e)) {   // the source assembles the blob (a GGUF read in place)
            s->jobs.push_back({nullptr, (size_t) lay.blob_bytes(layer), d.src, (int32_t) layer, e});
            if (ch && ch_n < 256) {
                ch_pairs[ch_n].first = (int32_t) layer;
                ch_pairs[ch_n].second = e;
                ++ch_n;
                ch_bytes += (uint64_t) lay.blob_bytes(layer);
            }
            continue;
        }
        const uint8_t* b = d.src->blob_stable(layer, e);
        if (b == nullptr) { err = "the expert source has no stable blob"; d.failed = true; return false; }
        s->jobs.push_back({b, (size_t) lay.blob_bytes(layer), nullptr, (int32_t) layer, e});
    }
    if (stager_fail_enabled() && !s->jobs.empty()) {
        // Controlled negative (diagnostic, default off): route EVERY source copy through the counter and make
        // the (ring+1)-th copy call fail (0-based cold-copy index ring+1) - by then the issuer has already
        // issued earlier jobs and the producer has crossed the host-ring wrap (jobs 0..kRing).  One bounded
        // case; the actual reached call/job and the issued count are reported by the stager stats.
        s->fault.real = d.src;
        s->fault.calls.store(0, std::memory_order_relaxed);
        s->fault.fail_at = (int) s->stager->kRing + 1;
        for (auto& jb : s->jobs)
            if (jb.from != nullptr) jb.from = &s->fault;
    }
    if (ch && ch_n > 0) {
        // Hand the current layer's transient pairs to the source ONCE, before start(), so the OS can begin the
        // three mapped role slices ahead of the producer's copies.  Success means the hint call was made (a
        // strong hint only - no residency/async guarantee); failure is ignored and the copy path is unchanged.
        const auto ch_t0 = std::chrono::steady_clock::now();
        const bool ch_ok = d.src->advise_pairs(ch_pairs, ch_n);
        s->ch_wall_us += (uint64_t) std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - ch_t0).count();
        ++s->ch_calls;
        if (ch_ok) ++s->ch_success;
        s->ch_pairs += ch_n;
        s->ch_bytes += ch_bytes;
    }
    const bool stager_on = !s->jobs.empty();
    if (stager_on) s->stager->start(std::move(s->jobs));
    struct LayerStagerDone {
        strata::prefill::ColdStager* st;
        ~LayerStagerDone() { if (st != nullptr) st->finish(); }
    } layer_stager_done{stager_on ? s->stager.get() : nullptr};
    std::vector<int32_t> bh, rel;
    for (int64_t j0 = 0; j0 < n_experts; j0 += kGroup) {
        const int64_t j1 = std::min<int64_t>(n_experts, j0 + kGroup);
        const int ngx = (int) (j1 - j0);
        int cold_slots[16];
        int n_cold = 0;
        int64_t max_cnt = 0;
        for (int64_t j = j0; j < j1; ++j) {
            const int32_t e = s->order[(size_t) j];
            max_cnt = std::max<int64_t>(max_cnt, s->cnt[(size_t) e]);
            const int32_t slot = d.cache != nullptr ? d.cache->slot_of(layer, e) : (int32_t) -1;
            if (slot >= 0) continue;
            const int rslot = s->ring_next++ % s->ring;
            // Only the DEVICE slot's reuse waits on the copy stream; the host ring belongs to the producer and
            // is protected by its own dma_done/issued protocol, so the old host-side guard on `used` went away
            // with ring_h.
            PhaseScope ph_ring_wait(ph, &s->ph_ringwait_ms);
            if (cudaStreamWaitEvent(s->copy, s->used[rslot], 0) != cudaSuccess) {
                err = "the prompt ring's reuse wait failed";
                return false;
            }
            cold_slots[n_cold] = rslot;
            ++n_cold;
        }
        PhaseScope ph_enq(ph, &s->ph_enqueue_ms);
        int cold_seen = 0;
        const uint8_t* mem_bd[kGroup];
        int mem_rslot[kGroup];
        for (int64_t j = j0; j < j1; ++j) {
            const int32_t e = s->order[(size_t) j];
            const int32_t slot = d.cache != nullptr ? d.cache->slot_of(layer, e) : (int32_t) -1;
            const uint8_t* bd = slot >= 0 ? d.cache->device_slot(slot) : nullptr;
            int rslot = -1;
            if (bd == nullptr) {
                rslot = cold_slots[cold_seen++];
                const int32_t job = s->job_of[(size_t) j];
                if (job < 0) { err = "the prompt staging job map is inconsistent"; d.failed = true; return false; }
                const uint8_t* hb;
                {
                    // The issuer waits for the producer's ready buffer - the phase that used to be the
                    // synchronous copy_blobs host wall; it is a different measurement now, not comparable.
                    PhaseScope ph_wait(ph, &s->ph_copyhost_ms);
                    hb = s->stager->wait(job);
                }
                if (hb == nullptr) {
                    err = std::string("the expert producer failed: ") + s->stager->fail_text();
                    d.failed = true;
                    return false;
                }
                if (stg_chk && s->stg_chk_n < kStagerChkMax) {
                    // Bounded producer byte check (diagnostic, default off): one host blob per distinct
                    // (gu_type, d_type), compared against a synchronous copy_blobs reference after the wait;
                    // the pair registers only after an equal compare.  A mismatch is a real failure.
                    const int32_t gu_t = (int32_t) f.gu_type;
                    const int32_t d_t = (int32_t) f.d_type;
                    bool known = false;
                    for (int i = 0; i < s->stg_chk_n; ++i)
                        if (s->stg_chk_pairs[i].gu_type == gu_t && s->stg_chk_pairs[i].d_type == d_t) { known = true; break; }
                    if (!known) {
                        const size_t bb = (size_t) lay.blob_bytes(layer);
                        std::vector<uint8_t> ref(bb);
                        int32_t l32 = (int32_t) layer, e32 = e;
                        uint8_t* ref_dst[1] = {ref.data()};
                        if (!d.src->copy_blobs(&l32, &e32, ref_dst, 1)) {
                            err = "the producer check's reference copy failed";
                            s->stg_chk_failed = true;
                            d.failed = true;
                            return false;
                        }
                        if (std::memcmp(hb, ref.data(), bb) != 0) {
                            std::fprintf(stderr, "strata qwen35: stager check MISMATCH gu_type %d d_type %d layer "
                                                 "%lld expert %d bytes %zu\n",
                                         gu_t, d_t, (long long) layer, e, bb);
                            err = "the producer check's byte compare failed";
                            s->stg_chk_failed = true;
                            d.failed = true;
                            return false;
                        }
                        StagerChkPair& rec = s->stg_chk_pairs[s->stg_chk_n++];
                        rec.gu_type = gu_t;
                        rec.d_type = d_t;
                        rec.layer = layer;
                        rec.expert = e;
                        rec.bytes = bb;
                        std::fprintf(stderr, "strata qwen35: stager check equal gu_type %d d_type %d layer %lld "
                                             "expert %d bytes %zu\n",
                                     gu_t, d_t, (long long) layer, e, bb);
                    }
                }
                if (cudaMemcpyAsync(s->ring_d[rslot], hb, (size_t) lay.blob_bytes(layer),
                                    cudaMemcpyHostToDevice, s->copy) != cudaSuccess) {
                    // Qwen's H2D failure has a callable entry: latch the producer, stop its writes, wake its
                    // waiters; the layer guard finishes it, the close path drains the stream.
                    s->stager->cancel("a cold expert's H2D enqueue failed");
                    err = "a cold expert's copy failed";
                    d.failed = true;
                    return false;
                }
                if (!s->stager->issued_one(job, s->copy)) {
                    err = std::string("the expert producer could not record a DMA: ") + s->stager->fail_text();
                    d.failed = true;
                    return false;
                }
                if (cudaEventRecord(s->copied[rslot], s->copy) != cudaSuccess) {
                    s->stager->cancel("a cold expert's copied-event record failed");
                    err = "a cold expert's copy failed";
                    d.failed = true;
                    return false;
                }
                bd = s->ring_d[rslot];
            }
            mem_bd[j - j0] = bd;
            mem_rslot[j - j0] = rslot;
        }
        if (n_cold > 0 && cudaStreamWaitEvent(cs, s->copied[cold_slots[n_cold - 1]], 0) != cudaSuccess) {
            err = "the prompt expert path's copy wait failed";
            return false;
        }
        // One gather_native_group launch moves the whole group's gate/up/down for the mixed resident/cold
        // members in the same j-j0 order (same up_off/half/down_off/stride parameters as the per-expert
        // gather).  Any 16-byte misalignment returns false BEFORE launching anything (moe_mmq.cu checks the
        // alignment first), and the original per-expert gather_native loop runs instead.  The release events
        // stay per-slot: each cold slot's own `used` is recorded after the group's reads are enqueued, so a
        // slot can only be overwritten once its gather completed.
        mmq::GatherGroup gg;
        const bool ev_gather = gpu_ev && ev_pair_begin(*s, cs, EV_GATHER);
        gg.first = 0;
        gg.n = (int) ngx;
        for (int q = 0; q < (int) ngx; ++q) gg.blob[q] = mem_bd[q];
        const bool grp_fast = mmq::gather_native_group(gg, f.up_off, s->gu_group / 2, f.down_off, s->d_group,
                                                       s->grp_gu, s->gu_group, s->grp_d, s->d_group, cs);
        if (grp_fast) {
            if (gg_stats) { ++s->gg_fast; ++s->gg_kernels; }
        } else {
            if (gg_stats) { ++s->gg_fallback; s->gg_kernels += ngx; }
            for (int q = 0; q < (int) ngx; ++q)
                mmq::gather_native(mem_bd[q], mem_bd[q] + f.up_off, s->gu_group / 2, mem_bd[q] + f.down_off,
                                   s->d_group, s->grp_gu + (size_t) q * s->gu_group,
                                   s->grp_d + (size_t) q * s->d_group, cs);
        }
        if (gpu_ev) s->ev_launch[EV_GATHER] += grp_fast ? 1 : ngx;
        if (ev_gather) ev_pair_end(*s, cs);
        for (int q = 0; q < (int) ngx; ++q) {
            if (mem_rslot[q] >= 0) {
                if (cudaEventRecord(s->used[mem_rslot[q]], cs) != cudaSuccess) {
                    err = "the prompt ring's release event failed";
                    return false;
                }
                if (gg_stats) ++s->gg_used_recs;
                ++s->streamed;
                s->stream_bytes += lay.blob_bytes(layer);
            } else {
                ++s->resident;
            }
        }
        if (gg_chk && grp_fast) {
            // Bounded byte check (STRATA_QWEN35_GATHER_CHECK=1): one check per distinct (gu_type, d_type)
            // actually taking the fast path in this allocation epoch - the first fast group of the pair is
            // re-gathered per expert into exact-size scratch and both destinations are compared over exactly
            // the bytes the gathers write (gate+up = 2 * gu_half at the slot start, down = d_bytes, slot by
            // slot).  A failed allocation, sync, copy or any mismatch is a REAL failure (err + d.failed;
            // scratch is freed by RAII); the pair registers only after a successful compare.  Default off:
            // no allocation, no sync, no clock, no flow change.
            const int32_t gu_t = (int32_t) f.gu_type;
            const int32_t d_t = (int32_t) f.d_type;
            bool chk_known = false;
            for (int i = 0; i < s->chk_n; ++i)
                if (s->chk_pairs[i].gu_type == gu_t && s->chk_pairs[i].d_type == d_t) { chk_known = true; break; }
            if (!chk_known) {
                if (s->chk_n >= kGatherChkMax) {
                    ++s->chk_skipped;
                } else {
                    const size_t gu_bytes = (size_t) ngx * s->gu_group;
                    const size_t d_bytes = (size_t) ngx * s->d_group;
                    const size_t gu_slot = 2 * (s->gu_group / 2);
                    const size_t d_slot = s->d_group;
                    GatherChkScratch scratch;
                    if (cudaMalloc(&scratch.gu, gu_bytes) != cudaSuccess ||
                        cudaMalloc(&scratch.d, d_bytes) != cudaSuccess) {
                        err = "the prompt group gather's byte check scratch could not be allocated";
                        s->chk_failed = true;
                        d.failed = true;
                        return false;
                    }
                    for (int q = 0; q < (int) ngx; ++q)
                        mmq::gather_native(mem_bd[q], mem_bd[q] + f.up_off, s->gu_group / 2, mem_bd[q] + f.down_off,
                                           s->d_group, (uint8_t*) scratch.gu + (size_t) q * s->gu_group,
                                           (uint8_t*) scratch.d + (size_t) q * s->d_group, cs);
                    if (cudaStreamSynchronize(cs) != cudaSuccess) {
                        err = "the prompt group gather's byte check sync failed";
                        s->chk_failed = true;
                        d.failed = true;
                        return false;
                    }
                    std::vector<uint8_t> a_gu(gu_bytes), b_gu(gu_bytes), a_d(d_bytes), b_d(d_bytes);
                    const bool c_ok =
                        cudaMemcpy(a_gu.data(), s->grp_gu, gu_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(b_gu.data(), scratch.gu, gu_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(a_d.data(), s->grp_d, d_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(b_d.data(), scratch.d, d_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
                    if (!c_ok) {
                        err = "the prompt group gather's byte check copy failed";
                        s->chk_failed = true;
                        d.failed = true;
                        return false;
                    }
                    size_t bad = SIZE_MAX;
                    int64_t bad_slot = -1;
                    const char* role = "";
                    for (int q = 0; q < (int) ngx && bad == SIZE_MAX; ++q) {
                        const size_t base = (size_t) q * s->gu_group;
                        for (size_t i = 0; i < gu_slot; ++i)
                            if (a_gu[base + i] != b_gu[base + i]) { bad = i; bad_slot = q; role = "gu"; break; }
                    }
                    for (int q = 0; q < (int) ngx && bad == SIZE_MAX; ++q) {
                        const size_t base = (size_t) q * s->d_group;
                        for (size_t i = 0; i < d_slot; ++i)
                            if (a_d[base + i] != b_d[base + i]) { bad = i; bad_slot = q; role = "down"; break; }
                    }
                    if (bad != SIZE_MAX) {
                        std::fprintf(stderr, "strata qwen35: gather check MISMATCH gu_type %d d_type %d layer %lld "
                                             "j0 %lld slot %lld role %s offset %zu gu_slot_bytes %zu "
                                             "d_slot_bytes %zu gu_bytes %zu d_bytes %zu\n",
                                     gu_t, d_t, (long long) layer, (long long) j0, (long long) bad_slot, role, bad,
                                     gu_slot, d_slot, gu_bytes, d_bytes);
                        err = "the prompt group gather's byte check failed";
                        s->chk_failed = true;
                        d.failed = true;
                        return false;
                    }
                    GatherChkPair& rec = s->chk_pairs[s->chk_n++];
                    rec.gu_type = gu_t;
                    rec.d_type = d_t;
                    rec.layer = layer;
                    rec.j0 = j0;
                    rec.ngx = (int32_t) ngx;
                    rec.gu_slot_bytes = gu_slot;
                    rec.d_slot_bytes = d_slot;
                    rec.gu_bytes = gu_bytes;
                    rec.d_bytes = d_bytes;
                    std::fprintf(stderr, "strata qwen35: gather check equal gu_type %d d_type %d layer %lld j0 %lld "
                                         "ngx %d gu_slot_bytes %zu d_slot_bytes %zu gu_bytes %zu d_bytes %zu (%s)\n",
                                 gu_t, d_t, (long long) layer, (long long) j0, (int) ngx, gu_slot, d_slot,
                                 gu_bytes, d_bytes,
                                 gu_slot == (size_t) s->gu_group ? "full slot incl. layout padding"
                                                                 : "gather-written extent only (unwritten gap excluded)");
                }
            }
        }
        bh.assign((size_t) (n_experts + 1), 0);
        for (int64_t j = 0; j < n_experts; ++j) bh[(size_t) j] = s->off[(size_t) s->order[(size_t) j]];
        bh[(size_t) n_experts] = (int32_t) n;
        rel.assign((size_t) ngx + 1, 0);
        for (int i = 0; i <= ngx; ++i)
            rel[(size_t) i] = bh[(size_t) std::min<int64_t>(n_experts, j0 + i)] - bh[(size_t) j0];
        const int64_t r0 = bh[(size_t) j0];
        const int64_t nr = bh[(size_t) j1] - r0;
        if (nr <= 0) continue;
        if (cudaMemcpyAsync(s->bounds, bh.data(), bh.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(s->rel, rel.data(), rel.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess) {
            err = "the prompt group bounds did not upload";
            return false;
        }
        if (cudaMemsetAsync(s->grp_gu + (size_t) ngx * s->gu_group, 0, kTail, cs) != cudaSuccess ||
            cudaMemsetAsync(s->grp_d + (size_t) ngx * s->d_group, 0, kTail, cs) != cudaSuccess) {
            err = "the prompt group's tail memset failed";
            return false;
        }
        mmq::Product gu;
        gu.w = s->grp_gu; gu.type = f.gu_type; gu.w_rows = 2 * f.n_ff; gu.w_cols = f.n_embd;
        gu.expert_bytes = s->gu_group; gu.n = ngx; gu.xq = s->xq; gu.bounds = s->bounds + j0; gu.ids = s->ident;
        gu.total_rows = n; gu.max_rows = max_cnt; gu.dst = s->gu; gu.ld_dst = 2 * f.n_ff;
        if (trace) std::fprintf(stderr, "qwen prefill trace: group %lld gu product (n %d r0 %lld nr %lld)\n", (long long) j0, ngx, (long long) r0, (long long) nr);
        const bool ev_gu = gpu_ev && ev_pair_begin(*s, cs, EV_GU);
        if (gpu_ev) ++s->ev_launch[EV_GU];
        s->ctx->run(gu, cs);
        if (ev_gu) ev_pair_end(*s, cs);
        if (trace && cudaDeviceSynchronize() != cudaSuccess)
            std::fprintf(stderr, "qwen prefill trace: GU PRODUCT FAULT\n");
        if (trace) std::fprintf(stderr, "qwen prefill trace: group %lld gu done\n", (long long) j0);
        ++s->product_calls;
        const bool ev_act = gpu_ev && ev_pair_begin(*s, cs, EV_ACT);
        if (gpu_ev) s->ev_launch[EV_ACT] += 2;   // swiglu + hidden quantize
        mmq::swiglu(s->gu + (size_t) r0 * 2 * f.n_ff, s->h + (size_t) r0 * f.n_ff, nr, f.n_ff, false, cs);
        mmq::quantize(s->h + (size_t) r0 * f.n_ff, nullptr, s->hq, f.d_type, f.n_ff, f.n_ff, nr, cs);
        if (ev_act) ev_pair_end(*s, cs);
        mmq::Product dn;
        dn.w = s->grp_d; dn.type = f.d_type; dn.w_rows = f.n_embd; dn.w_cols = f.n_ff;
        dn.expert_bytes = s->d_group; dn.n = ngx; dn.xq = s->hq; dn.bounds = s->rel; dn.ids = s->ident;
        dn.total_rows = nr; dn.max_rows = max_cnt; dn.dst = s->dm + (size_t) r0 * f.n_embd; dn.ld_dst = f.n_embd;
        const bool ev_dn = gpu_ev && ev_pair_begin(*s, cs, EV_DN);
        if (gpu_ev) ++s->ev_launch[EV_DN];
        s->ctx->run(dn, cs);
        if (ev_dn) ev_pair_end(*s, cs);
        if (trace && cudaDeviceSynchronize() != cudaSuccess)
            std::fprintf(stderr, "qwen prefill trace: DOWN PRODUCT FAULT\n");
        if (trace) std::fprintf(stderr, "qwen prefill trace: group %lld down done\n", (long long) j0);
        ++s->product_calls;
    }
    {
    PhaseScope ph_combine(ph, &s->ph_combine_ms);
    const bool ev_combine = gpu_ev && ev_pair_begin(*s, cs, EV_COMBINE);
    if (gpu_ev) ++s->ev_launch[EV_COMBINE];
    if (trace) std::fprintf(stderr, "qwen prefill trace: combine launch\n");
    combine_scatter_kernel<<<dim3((unsigned) T), kThreads, 0, cs>>>(
        s->dm, s->weights_dev, s->entry_of_tok, (float*) out_dev, f.n_embd, (int) k);
    if (ev_combine) ev_pair_end(*s, cs);
    if (trace) {
        cudaStreamSynchronize(cs);
        float h1[4], h2[4], h3[4], h4[4];
        int32_t e4[8];
        {
            std::vector<float> all((size_t) T * f.n_embd);
            cudaMemcpy(all.data(), out_dev, all.size() * sizeof(float), cudaMemcpyDeviceToHost);
            int64_t bad = 0;
            for (float v : all) if (!std::isfinite(v)) ++bad;
            int64_t bad_dm = 0;
            std::vector<float> dmh((size_t) s->T_cap * f.n_embd);
            cudaMemcpy(dmh.data(), s->dm, dmh.size() * sizeof(float), cudaMemcpyDeviceToHost);
            for (float v : dmh) if (!std::isfinite(v)) ++bad_dm;
            std::fprintf(stderr, "qwen prefill trace: layer %lld nonfinite out %lld of %lld, dm %lld\n",
                         (long long) layer, (long long) bad, (long long) all.size(), (long long) bad_dm);
        }
        cudaMemcpy(h1, s->dm, sizeof(h1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h2, s->gu, sizeof(h2), cudaMemcpyDeviceToHost);
        cudaMemcpy(h3, s->weights_dev, sizeof(h3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h4, out_dev, sizeof(h4), cudaMemcpyDeviceToHost);
        cudaMemcpy(e4, s->entry_of_tok, sizeof(e4), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "qwen prefill trace: dm %g %g %g %g | gu %g %g %g %g | w %g %g %g %g | out %g %g %g %g"
                             " | eot %d %d %d %d %d %d %d %d\n", h1[0], h1[1], h1[2], h1[3], h2[0], h2[1], h2[2],
                     h2[3], h3[0], h3[1], h3[2], h3[3], h4[0], h4[1], h4[2], h4[3], e4[0], e4[1], e4[2], e4[3],
                     e4[4], e4[5], e4[6], e4[7]);
    }
    if (cudaGetLastError() != cudaSuccess) { err = "the prompt expert combine launch failed"; return false; }
    }
    // One-shot single-layer capture (STRATA_QWEN35_LAYER_CAPTURE=<dir>, default off): the first T==2048 call
    // of layer 16 copies the call's real x/ids/weights, the reference combine output and the after-loan
    // residency mask (slot per expert), with ONE sync on this work stream for the captured layer only.  That
    // extra sync perturbs this call by construction; the capture is never a speed measurement.
    if (s->cap_this_call) {
        const int64_t CH = f.n_embd, CK = k, CNE = d.n_expert;
        std::vector<float> ch_x((size_t) T * CH), ch_w((size_t) T * CK), ch_o((size_t) T * CH);
        std::vector<int32_t> ch_id((size_t) T * CK), ch_mask((size_t) CNE, -1);
        bool cok = true;
        for (int64_t t = 0; t < T && cok; ++t)
            cok = cudaMemcpyAsync(ch_x.data() + (size_t) t * CH, (const uint8_t*) x_dev + (size_t) t * x_nb1,
                                  (size_t) CH * sizeof(float), cudaMemcpyDeviceToHost, cs) == cudaSuccess;
        for (int64_t t = 0; t < T && cok; ++t) {
            cok = cudaMemcpyAsync(ch_id.data() + (size_t) t * CK, (const uint8_t*) ids_dev + (size_t) t * ids_nb1,
                                  (size_t) CK * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) == cudaSuccess;
            if (cok)
                cok = cudaMemcpy2DAsync(ch_w.data() + (size_t) t * CK, sizeof(float),
                                        (const uint8_t*) weights_dev + (size_t) t * w_nb2, (size_t) w_nb1,
                                        sizeof(float), (size_t) CK, cudaMemcpyDeviceToHost, cs) == cudaSuccess;
        }
        if (cok)
            cok = cudaMemcpyAsync(ch_o.data(), out_dev, (size_t) T * CH * sizeof(float),
                                  cudaMemcpyDeviceToHost, cs) == cudaSuccess;
        if (cok) cok = cudaStreamSynchronize(cs) == cudaSuccess;
        if (!cok) { err = "the layer capture's copies did not complete"; d.failed = true; return false; }
        int64_t x_nf = 0, w_nf = 0, o_nf = 0, ids_bad = 0;
        for (float v : ch_x) if (!std::isfinite(v)) ++x_nf;
        for (float v : ch_w) if (!std::isfinite(v)) ++w_nf;
        for (float v : ch_o) if (!std::isfinite(v)) ++o_nf;
        for (int32_t v : ch_id) if (v < 0 || v >= CNE) ++ids_bad;
        int64_t mask_res = 0;
        for (int64_t e = 0; e < CNE; ++e) {
            ch_mask[(size_t) e] = d.cache != nullptr ? d.cache->slot_of((int64_t) layer, e) : -1;
            if (ch_mask[(size_t) e] >= 0) ++mask_res;
        }
        if (x_nf != 0 || w_nf != 0 || o_nf != 0 || ids_bad != 0) {
            err = "the layer capture found non-finite values or out-of-range ids";
            d.failed = true;
            return false;
        }
        const uint64_t xh = fnv_bytes(kFnvSeed, ch_x.data(), ch_x.size() * sizeof(float));
        const uint64_t ih = fnv_bytes(kFnvSeed, ch_id.data(), ch_id.size() * sizeof(int32_t));
        const uint64_t whe = fnv_bytes(kFnvSeed, ch_w.data(), ch_w.size() * sizeof(float));
        const uint64_t oh = fnv_bytes(kFnvSeed, ch_o.data(), ch_o.size() * sizeof(float));
        const char* exe_sha = std::getenv("STRATA_QWEN35_CAPTURE_EXE_SHA");
        auto wfile = [&](const char* name, const void* p, size_t bytes) -> bool {
            std::string path = std::string(cap_dir) + "/" + name;
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            return out && out.write((const char*) p, (std::streamsize) bytes);
        };
        bool wok = wfile("x.f32", ch_x.data(), ch_x.size() * sizeof(float)) &&
                   wfile("ids.i32", ch_id.data(), ch_id.size() * sizeof(int32_t)) &&
                   wfile("weights.f32", ch_w.data(), ch_w.size() * sizeof(float)) &&
                   wfile("out.f32", ch_o.data(), ch_o.size() * sizeof(float)) &&
                   wfile("mask.i32", ch_mask.data(), ch_mask.size() * sizeof(int32_t));
        std::ofstream meta(std::string(cap_dir) + "/meta.txt", std::ios::trunc);
        wok = wok && (bool) meta;
        if (wok) {
            meta << "layer " << (long long) layer << "\n"
                 << "T " << (long long) T << "\n"
                 << "K " << (long long) CK << "\n"
                 << "H " << (long long) CH << "\n"
                 << "FF " << (long long) f.n_ff << "\n"
                 << "n_layers " << (long long) lay.fmt.size() << "\n"
                 << "n_expert " << (long long) CNE << "\n"
                 << "gu_type " << (int) f.gu_type << "\n"
                 << "d_type " << (int) f.d_type << "\n"
                 << "up_off " << (long long) f.up_off << "\n"
                 << "down_off " << (long long) f.down_off << "\n"
                 << "x_nb1 " << (long long) x_nb1 << "\n"
                 << "ids_nb1 " << (long long) ids_nb1 << "\n"
                 << "w_nb1 " << (long long) w_nb1 << "\n"
                 << "w_nb2 " << (long long) w_nb2 << "\n"
                 << "blob_bytes " << (unsigned long long) lay.blob_bytes((int64_t) layer) << "\n"
                 << "exe_sha " << (exe_sha != nullptr ? exe_sha : "unknown") << "\n"
                 << "perturbed 1\n"
                 << "mask_resident " << (long long) mask_res << "\n"
                 << "x_fnv 0x" << std::hex << xh << std::dec << "\n"
                 << "ids_fnv 0x" << std::hex << ih << std::dec << "\n"
                 << "w_fnv 0x" << std::hex << whe << std::dec << "\n"
                 << "out_fnv 0x" << std::hex << oh << std::dec << "\n";
            wok = (bool) meta;
        }
        if (!wok) { err = "the layer capture could not write its files"; d.failed = true; return false; }
        s->cap_done = true;
        s->cap_this_call = false;
        std::fprintf(stderr, "strata qwen35: layer capture (STRATA_QWEN35_LAYER_CAPTURE): layer %lld T %lld K %lld "
                             "H %lld mask_resident %lld x_nf %lld w_nf %lld out_nf %lld ids_bad %lld dir %s\n",
                     (long long) layer, (long long) T, (long long) CK, (long long) CH, (long long) mask_res,
                     (long long) x_nf, (long long) w_nf, (long long) o_nf, (long long) ids_bad, cap_dir);
    }
    // Route probe B (default off): the combine output is read AFTER the combine on the same work stream with
    // ONE bounded sync (before the bridge records `post`, which stays stream-ordered after this).  Then every
    // captured array is hashed and written: raw H*T f32, no padding, real values only.
    if (probe && (probe_a || probe_c || probe_b)) {
        const size_t cap = (size_t) T * (size_t) f.n_embd * sizeof(float);
        const size_t row = (size_t) f.n_embd * sizeof(float);
        if (probe_b) {
            if (cudaMemcpyAsync(s->probe_buf.data() + 2 * cap, out_dev, cap, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                cudaStreamSynchronize(cs) != cudaSuccess) {
                err = "the route probe's output copy failed";
                d.failed = true;
                return false;
            }
        }
        const char* dir = std::getenv("STRATA_QWEN35_ROUTE_PROBE_DIR");
        if (dir == nullptr || dir[0] == '\0') {
            err = "the route probe has no output directory (STRATA_QWEN35_ROUTE_PROBE_DIR)";
            d.failed = true;
            return false;
        }
        const struct {
            const char* tag;
            bool want;
            size_t off;
            int64_t layer;
            size_t stride;
        } probe_arr[3] = {{"A", probe_a, 0, (int64_t) layer, (size_t) x_nb1},
                          {"C", probe_c, cap, (int64_t) layer, (size_t) x_nb1},
                          {"B", probe_b, 2 * cap, (int64_t) layer, row}};
        for (const auto& pa : probe_arr) {
            if (!pa.want) continue;
            const uint8_t* p = s->probe_buf.data() + pa.off;
            int64_t finite = 0;
            for (size_t i = 0; i + sizeof(float) <= cap; i += sizeof(float)) {
                float v;
                std::memcpy(&v, p + i, sizeof(v));
                if (std::isfinite(v)) ++finite;
            }
            const uint64_t h = fnv_bytes(kFnvSeed, p, cap);
            std::fprintf(stderr, "strata qwen35: route probe (STRATA_QWEN35_ROUTE_PROBE=1): array %s layer %lld "
                                 "call %lld occ 2 T %lld H %lld stride %zu bytes %zu finite %lld hash 0x%016llx\n",
                         pa.tag, (long long) pa.layer, (long long) probe_call, (long long) T,
                         (long long) f.n_embd, pa.stride, cap, (long long) finite,
                         (unsigned long long) h);
            std::string path = std::string(dir) + "/probe-" + pa.tag + "-layer" + std::to_string(pa.layer) +
                               "-occ2.f32";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out || !out.write((const char*) p, (std::streamsize) cap)) {
                err = std::string("the route probe could not write ") + path;
                d.failed = true;
                return false;
            }
        }
    }
    d.native_stats.prefill_layers_gpu += 1;
    d.native_stats.prefill_resident = s->resident;
    d.native_stats.prefill_streamed = s->streamed;
    d.native_stats.prefill_product_calls = s->product_calls;
    d.native_stats.prefill_stream_bytes = s->stream_bytes;
    d.native_stats.prefill_workspace_bytes = s->workspace_bytes;
    d.native_stats.prefill_ring_bytes = s->ring_bytes;
    d.native_stats.prefill_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool qwen_prefill_loan_return(ExpertDispatch& d, std::string& err) {
    PrefillState* s = state(d);
    if (s == nullptr || s->loan_returned) return true;
    if (!d.native_loan_active) {
        // Nothing is lent (owned buffers, or a loan already given back): only now is the state final.
        s->loan_returned = true;
        return true;
    }
    // The prompt's device work holds the borrowed region; the lent slots may only be refilled once it is done.
    // `loan_returned` is set ONLY after a real success: a failed sync/refill leaves the state retryable (the
    // close path tries again) and the slots not-resident, never "succeeded by flag".
    const cudaError_t de = cudaDeviceSynchronize();
    if (de != cudaSuccess) {
        err = std::string("the prompt path's device work did not finish before the loan return: ") +
              cudaGetErrorString(de);
        return false;
    }
    if (!expert_native_loan_return(d, err)) return false;
    s->loan_returned = true;
    return true;
}

void qwen_prefill_close(ExpertDispatch& d) {
    PrefillState* s = state(d);
    if (s == nullptr) return;
    std::string err;
    if (!qwen_prefill_loan_return(d, err)) {
        std::fprintf(stderr, "strata qwen35: the prompt loan could not be returned: %s\n", err.c_str());
        // The return bailed (e.g. its device sync, or the refill itself, failed).  expert_native_loan_return
        // only syncs its own refill queue - it gives no device-wide drain for the prompt copy/compute - so the
        // refill must not run on an unconfirmed device.  ONE explicit best-effort CHECKED fallback:
        const cudaError_t fb = cudaDeviceSynchronize();
        if (fb != cudaSuccess) fatal_drain_unconfirmed("the prompt loan return", fb);
        std::fprintf(stderr, "strata qwen35: fallback checked device sync succeeded; retrying the loan return\n");
        std::string e2;
        if (!expert_native_loan_return(d, e2)) {
            // Drain confirmed, but the refill/verify did not succeed: the slots stay NOT resident and the loan
            // is not marked returned; ownership of the (never-reused) borrowed region stays with the cache.
            std::fprintf(stderr, "strata qwen35: the fallback loan return failed: %s; the slots stay not-resident\n",
                         e2.c_str());
        } else {
            s->loan_returned = true;
        }
    }
    if (route_fp_enabled() && s->fp_calls > 0) {
        // One final block (no per-layer stdio during the prompt).  Rules: call = prefill-call order; ids_hash =
        // FNV-1a over the T*k int32 ids in (token, rank) order; weights_hash = FNV-1a over the same-order raw
        // float bits; distinct/groups/resident/streamed are this call's host-side counts (groups = non-empty
        // kGroup groups; resident/streamed from slot_of, the same decision the group loop makes); lent_hash =
        // FNV-1a over the loan's (residency index, slot) pairs.
        std::fprintf(stderr, "strata qwen35: route fingerprint (STRATA_QWEN35_ROUTE_FP=1): calls %lld total 0x%016llx\n",
                     (long long) s->fp_calls, (unsigned long long) s->fp_total);
        std::fprintf(stderr, "strata qwen35: route fp loan: slots %lld first %lld bytes %llu lent_hash 0x%016llx\n",
                     (long long) s->fp_loan_slots, (long long) s->fp_loan_first,
                     (unsigned long long) s->fp_loan_bytes, (unsigned long long) s->fp_loan_hash);
        std::fprintf(stderr, "strata qwen35: route fp rec: call layer T k distinct groups resident streamed ids_hash weights_hash\n");
        for (const RouteFpRec& r : s->fp_recs)
            std::fprintf(stderr, "route_fp %lld %lld %lld %lld %lld %lld %lld %lld 0x%016llx 0x%016llx\n",
                         (long long) r.call, (long long) r.layer, (long long) r.T, (long long) r.k,
                         (long long) r.distinct, (long long) r.groups, (long long) r.resident,
                         (long long) r.streamed, (unsigned long long) r.ids_hash,
                         (unsigned long long) r.weights_hash);
    }
    if (prefill_phases_enabled() && s->ph_calls > 0) {
        // Per-call host wall sums at EXISTING boundaries (no new CUDA events/syncs).  Definitions: d2h_sync =
        // the two original strided D2H enqueues + the one work-stream sync (the wait includes the upstream
        // graph's readiness, so it is NOT pure copy time); grouping = weights unpack + weights_dev upload
        // enqueue + id grouping + row maps (host); ring_wait = the copy-stream reuse wait for a device ring
        // slot (the old host-side guard is gone with ring_h); copy_host = the ISSUER waiting for the producer's
        // ready buffer (this used to be the synchronous copy_blobs host wall - different measurements, do not
        // compare them head-to-head); enqueue = the group walk's H2D/event enqueues plus the device section
        // (bounds memsets/uploads, gather, MMQ products, swiglu, quantize) as HOST-SIDE launch API time, not
        // GPU kernel time (it contains the producer waits); combine = the combine launch + its error check.
        // Sums overlap the existing duration counters and must not be added together as a critical path.
        std::fprintf(stderr, "strata qwen35: prefill phases (STRATA_QWEN35_PREFILL_PHASES=1): calls %lld\n",
                     (long long) s->ph_calls);
        std::fprintf(stderr, "strata qwen35: prefill phases ms: d2h_sync %.2f | grouping %.2f | ring_wait %.2f | "
                             "copy_host %.2f | enqueue %.2f | combine %.2f\n",
                     s->ph_d2h_ms, s->ph_group_ms, s->ph_ringwait_ms, s->ph_copyhost_ms, s->ph_enqueue_ms,
                     s->ph_combine_ms);
        std::fprintf(stderr, "strata qwen35: prefill phases note: d2h_sync includes upstream readiness (not pure "
                             "copy); enqueue is host launch API time (not kernel time).\n");
    }
    if (gather_stats_enabled() && (s->gg_fast > 0 || s->gg_fallback > 0)) {
        // Definitions (this build): groups_fast = groups whose gate/up/down moved in ONE gather_native_group
        // launch; groups_fallback = groups that took the per-expert gather_native loop (16B alignment false,
        // nothing launched by the group call); kernel_launches = 1 per fast group + 1 per expert in fallback
        // groups; used_records = per-cold-slot cudaEventRecord(used).  prefill_product_calls is a different
        // counter (2 MMQ products per group) and is not changed by the gather.
        std::fprintf(stderr, "strata qwen35: gather stats (STRATA_QWEN35_GATHER_STATS=1): groups_fast %lld "
                             "groups_fallback %lld kernel_launches %lld used_records %lld\n",
                     (long long) s->gg_fast, (long long) s->gg_fallback, (long long) s->gg_kernels,
                     (long long) s->gg_used_recs);
    }
    if (gather_check_enabled()) {
        // The pair registry is per allocation epoch (shape); each distinct (gu_type, d_type) is checked at
        // its FIRST fast group only.  Pairs the prompt's routing never hit are NOT tested and are listed
        // against the layout's distinct pairs, so partial coverage cannot read as a full pass.
        int layout_pairs = 0;
        int32_t l_gu[kGatherChkMax] = {}, l_d[kGatherChkMax] = {};
        for (const auto& f2 : strata::kernels::cpu::expert_layout().fmt) {
            bool seen = false;
            for (int i = 0; i < layout_pairs; ++i)
                if (l_gu[i] == f2.gu_type && l_d[i] == f2.d_type) { seen = true; break; }
            if (!seen && layout_pairs < kGatherChkMax) {
                l_gu[layout_pairs] = f2.gu_type;
                l_d[layout_pairs] = f2.d_type;
                ++layout_pairs;
            }
        }
        std::fprintf(stderr, "strata qwen35: gather check pairs (STRATA_QWEN35_GATHER_CHECK=1): checked %d of %d "
                             "distinct layout (gu_type,d_type) pairs\n",
                     s->chk_n, layout_pairs);
        for (int i = 0; i < s->chk_n; ++i) {
            const GatherChkPair& p = s->chk_pairs[i];
            std::fprintf(stderr, "strata qwen35: gather check pair checked[%d] gu_type %d d_type %d layer %lld j0 "
                                 "%lld ngx %d gu_slot_bytes %zu d_slot_bytes %zu gu_bytes %zu d_bytes %zu equal\n",
                         i, p.gu_type, p.d_type, (long long) p.layer, (long long) p.j0, p.ngx, p.gu_slot_bytes,
                         p.d_slot_bytes, p.gu_bytes, p.d_bytes);
        }
        if (s->chk_failed)
            std::fprintf(stderr, "strata qwen35: gather check pairs: FAILED (a requested check did not complete; "
                                 "see the error above; the failing pair is NOT registered)\n");
        if (s->chk_skipped > 0)
            std::fprintf(stderr, "strata qwen35: gather check pairs: %d fast group(s) of unregistered pairs "
                                 "skipped beyond the %d-pair diagnostic limit (NOT checked)\n",
                         s->chk_skipped, kGatherChkMax);
        for (int i = 0; i < layout_pairs; ++i) {
            bool chk = false;
            for (int j = 0; j < s->chk_n; ++j)
                if (s->chk_pairs[j].gu_type == l_gu[i] && s->chk_pairs[j].d_type == l_d[i]) { chk = true; break; }
            if (!chk)
                std::fprintf(stderr, "strata qwen35: gather check pair NOT checked gu_type %d d_type %d "
                                     "(%s)\n",
                             l_gu[i], l_d[i],
                             s->chk_failed ? "an earlier check failed"
                                           : "not encountered by this prompt's fast groups");
        }
    }
    if (s->stager != nullptr && stager_stats_enabled()) {
        // Sparse producer summary (STRATA_QWEN35_STAGER_STATS=1, default off): starts = layers that started a
        // job list; jobs = staged cold experts; issued = dma_done records the issuer made (one per cold job,
        // in order); finishes = layer guards run; failures = first cancels latched; active = producers inside
        // a job (0 after every finish); pin = the producer's real cudaHostAlloc bytes (kRing x blob, strict).
        std::fprintf(stderr, "strata qwen35: stager stats (STRATA_QWEN35_STAGER_STATS=1): starts %llu jobs %llu "
                             "issued %llu finishes %llu failures %llu active %d ring %d batch %d sleep %d "
                             "strict_pin %d propagate_error %d host_pin_bytes %zu\n",
                     (unsigned long long) s->stager->stat_starts.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->stat_jobs.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->stat_issued.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->stat_finishes.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->stat_fails.load(std::memory_order_relaxed),
                     s->stager->active_now(), s->stager->kRing, s->stager->batch, s->stager->sleep_mode ? 1 : 0,
                     s->stager->strict_pin ? 1 : 0, s->stager->opt ? 1 : 0, s->stager->host_pinned_bytes());
        // The actual chosen producer count (1 default; 2/3 only via STRATA_QWEN35_STAGER_THREADS="2"/"3").
        std::fprintf(stderr, "strata qwen35: stager workers (STRATA_QWEN35_STAGER_STATS=1): chosen %d\n",
                     s->stager_threads);
        if (s->stager->failed())
            std::fprintf(stderr, "strata qwen35: stager failure text: %s\n", s->stager->fail_text());
        // Producer-side host-wall scopes (opt-in with STAGER_STATS): the scopes are per WORKER and overlap across
        // workers, so with two producers the sums must only be read as a sum (not wall clock, not CPU busy);
        // wakes/first_claim are per worker-generation and need not equal the layer count.  They are HOST API wall
        // only (a fast cudaEventSynchronize or issued check still contributes), NOT GPU busy time and not additive
        // to the GPU critical path.  claim/ready/issued_one and the between-layer cv wait are not covered.  A
        // failed/cancelled copy contributes wall but never success jobs/bytes.
        std::fprintf(stderr, "strata qwen35: stager perf (STRATA_QWEN35_STAGER_STATS=1): src_copy_us %llu "
                             "(jobs %llu bytes %llu) | issued_wait_us %llu (entries %llu) | dma_wait_us %llu "
                             "(calls %llu) | wakes %llu first_claim_us %llu (count %llu)\n",
                     (unsigned long long) s->stager->perf_src_us.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_src_jobs.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_src_bytes.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_issued_us.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_issued_entries.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_dma_us.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_dma_calls.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_wakes.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_first_us.load(std::memory_order_relaxed),
                     (unsigned long long) s->stager->perf_first_n.load(std::memory_order_relaxed));
    }
    if (cold_hint_enabled() && s->ch_calls > 0) {
        // Bounded per-request aggregate (hint-on only).  success = advise_pairs calls that actually made the OS
        // hint (PrefetchVirtualMemory on Windows): a STRONG HINT does not prove the pages are resident, async
        // or a speed gain; the wall is measured inside the prefill total, not moved outside the timing.
        std::fprintf(stderr, "strata qwen35: cold hint (STRATA_QWEN35_COLD_HINT=1): calls %lld success %lld pairs "
                             "%lld bytes %llu wall_us %llu (strong hint only: no residency/async/speed guarantee)\n",
                     (long long) s->ch_calls, (long long) s->ch_success, (long long) s->ch_pairs,
                     (unsigned long long) s->ch_bytes, (unsigned long long) s->ch_wall_us);
    }
    if (stager_check_enabled()) {
        // Bounded producer byte check (one host blob per actual layer-format pair; registered after an equal
        // compare).  Unencountered pairs are not tested and the failure state is reported, never a pass.
        std::fprintf(stderr, "strata qwen35: stager check pairs (STRATA_QWEN35_STAGER_CHECK=1): checked %d\n",
                     s->stg_chk_n);
        for (int i = 0; i < s->stg_chk_n; ++i) {
            const StagerChkPair& p = s->stg_chk_pairs[i];
            std::fprintf(stderr, "strata qwen35: stager check pair checked[%d] gu_type %d d_type %d layer %lld "
                                 "expert %d bytes %zu equal\n",
                         i, p.gu_type, p.d_type, (long long) p.layer, p.expert, p.bytes);
        }
        if (s->stg_chk_failed)
            std::fprintf(stderr, "strata qwen35: stager check pairs: FAILED (a requested check did not complete; "
                                 "the failing pair is NOT registered)\n");
    }
    // GPU-event diagnostic tail (default off): the last call's pairs may still be pending; ONE wait on the last
    // recorded event completes them (never a per-kernel sync).  An unconfirmable tail is reported as dropped
    // and as UNKNOWN, never as a completed measurement.
    if (s->evt_ready && s->evt_pairs > 0) {
        const cudaError_t se = cudaEventSynchronize(s->evt[2 * s->evt_pairs - 1]);
        if (se == cudaSuccess) {
            ev_collect(*s);
        } else {
            s->ev_pending_unknown = true;
            s->ev_disabled = true;
            ++s->ev_calls_failed;
            s->ev_dropped += s->evt_pairs;
            std::fprintf(stderr, "strata qwen35: gpu events: the last recorded event never completed (%s); "
                                 "%d pending pair(s) DROPPED and the tail measurement is UNKNOWN\n",
                         cudaGetErrorString(se), s->evt_pairs);
            s->evt_pairs = 0;
        }
    }
    if (gpu_events_enabled() &&
        (s->ev_calls > 0 || s->ev_calls_failed > 0 || s->ev_phase_expected[EV_XQ] > 0 || s->ev_create_failed > 0)) {
        const double sum = s->ev_phase_ms[EV_XQ] + s->ev_phase_ms[EV_GATHER] + s->ev_phase_ms[EV_GU] +
                           s->ev_phase_ms[EV_ACT] + s->ev_phase_ms[EV_DN] + s->ev_phase_ms[EV_COMBINE];
        const bool unknown = s->ev_calls_failed > 0 || s->ev_span_fail > 0 || s->ev_truncated > 0 ||
                             s->ev_record_failed > 0 || s->ev_create_failed > 0 || s->ev_dropped > 0 ||
                             s->ev_pending_unknown || s->ev_disabled;
        std::fprintf(stderr, "strata qwen35: gpu events (STRATA_QWEN35_GPU_EVENTS=1): calls_ok %lld calls_failed %lld "
                             "span_ms %.3f xq_ms %.3f gather_ms %.3f gu_ms %.3f act_ms %.3f dn_ms %.3f combine_ms %.3f "
                             "other_ms %.3f status %s\n",
                     (long long) s->ev_calls, (long long) s->ev_calls_failed, s->ev_phase_ms[EV_SPAN],
                     s->ev_phase_ms[EV_XQ], s->ev_phase_ms[EV_GATHER], s->ev_phase_ms[EV_GU],
                     s->ev_phase_ms[EV_ACT], s->ev_phase_ms[EV_DN], s->ev_phase_ms[EV_COMBINE],
                     s->ev_phase_ms[EV_SPAN] - sum, unknown ? "UNKNOWN (missing measurements)" : "COMPLETE");
        std::fprintf(stderr, "strata qwen35: gpu events phases ok/expected: xq %lld/%lld gather %lld/%lld gu %lld/%lld "
                             "act %lld/%lld dn %lld/%lld combine %lld/%lld; elapsed_fail xq %lld gather %lld gu %lld "
                             "act %lld dn %lld combine %lld span_fail %lld\n",
                     (long long) s->ev_phase_ok[EV_XQ], (long long) s->ev_phase_expected[EV_XQ],
                     (long long) s->ev_phase_ok[EV_GATHER], (long long) s->ev_phase_expected[EV_GATHER],
                     (long long) s->ev_phase_ok[EV_GU], (long long) s->ev_phase_expected[EV_GU],
                     (long long) s->ev_phase_ok[EV_ACT], (long long) s->ev_phase_expected[EV_ACT],
                     (long long) s->ev_phase_ok[EV_DN], (long long) s->ev_phase_expected[EV_DN],
                     (long long) s->ev_phase_ok[EV_COMBINE], (long long) s->ev_phase_expected[EV_COMBINE],
                     (long long) s->ev_phase_fail[EV_XQ], (long long) s->ev_phase_fail[EV_GATHER],
                     (long long) s->ev_phase_fail[EV_GU], (long long) s->ev_phase_fail[EV_ACT],
                     (long long) s->ev_phase_fail[EV_DN], (long long) s->ev_phase_fail[EV_COMBINE],
                     (long long) s->ev_span_fail);
        std::fprintf(stderr, "strata qwen35: gpu events launches (kernel enqueues): xq %lld gather %lld gu %lld act %lld "
                             "dn %lld combine %lld; truncated_pairs %lld record_failed %lld create_failed %lld dropped %lld "
                             "pending_unknown %d\n",
                     (long long) s->ev_launch[EV_XQ], (long long) s->ev_launch[EV_GATHER],
                     (long long) s->ev_launch[EV_GU], (long long) s->ev_launch[EV_ACT],
                     (long long) s->ev_launch[EV_DN], (long long) s->ev_launch[EV_COMBINE],
                     (long long) s->ev_truncated, (long long) s->ev_record_failed,
                     (long long) s->ev_create_failed, (long long) s->ev_dropped, s->ev_pending_unknown ? 1 : 0);
        std::fprintf(stderr, "strata qwen35: gpu events note: every phase span is a STREAM elapsed (it includes "
                             "launch gaps and stream waits enqueued between the pair), NOT SM/kernel busy and not "
                             "summable into a full critical path; a gather pair spans 1 launch (fast group path) or "
                             "ngx launches (fallback loop), so its pair count is NOT its launch count; pairs collect "
                             "at the next call's D2H sync and the tail after one last-event wait\n");
    }
    if (rowhist_enabled() && s->rh_layers > 0) {
        std::fprintf(stderr, "strata qwen35: row hist (STRATA_QWEN35_ROWHIST=1): layers %lld buckets 1:%lld "
                             "2-3:%lld 4-7:%lld 8-15:%lld 16-31:%lld 32-63:%lld 64-127:%lld 128-255:%lld "
                             "256-511:%lld 512-1023:%lld 1024-2047:%lld 2048+:%lld\n",
                     (long long) s->rh_layers, (long long) s->rh_buckets[0], (long long) s->rh_buckets[1],
                     (long long) s->rh_buckets[2], (long long) s->rh_buckets[3], (long long) s->rh_buckets[4],
                     (long long) s->rh_buckets[5], (long long) s->rh_buckets[6], (long long) s->rh_buckets[7],
                     (long long) s->rh_buckets[8], (long long) s->rh_buckets[9], (long long) s->rh_buckets[10],
                     (long long) s->rh_buckets[11]);
        for (size_t il = 0; il < s->rh_distinct.size(); ++il) {
            if (s->rh_distinct[il] <= 0) continue;
            std::fprintf(stderr, "rowhist layer %zu distinct %lld maxcnt %lld rows %lld groups %lld\n", il,
                         (long long) s->rh_distinct[il], (long long) s->rh_maxcnt[il],
                         (long long) s->rh_rows[il], (long long) s->rh_groups[il]);
        }
    }
    // The producer's host writes are already stopped (every layer's guard ran finish() and waited active==0);
    // the copy stream must now be CHECKED idle before free_all releases the pinned ring and destroys the
    // stager.  One explicit best-effort checked fallback; if the drain stays unconfirmable, stop here: the
    // loan is not refilled and nothing is released (the fatal rule above).
    if (s->copy != nullptr) {
        const cudaError_t ce = cudaStreamSynchronize(s->copy);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: the prompt copy stream did not drain: %s\n", cudaGetErrorString(ce));
            const cudaError_t fb = cudaDeviceSynchronize();
            if (fb != cudaSuccess) fatal_drain_unconfirmed("the prompt workspace release", fb);
            std::fprintf(stderr, "strata qwen35: fallback checked device sync succeeded; releasing the workspace\n");
        }
    }
    free_all(*s);
    delete s;
    d.prefill_state = nullptr;
}

}  // namespace strata::core
