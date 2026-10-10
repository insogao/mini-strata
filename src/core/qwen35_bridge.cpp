// src/core/qwen35_bridge.cpp - the Qwen35 routed-expert boundary.  See the header.
//
// THE BOUNDARY IS THIN ON PURPOSE.  It owns only the graph-level pieces: the CUSTOM node for the model graph,
// the scheduler's external-producer callback, and the D2H/H2D of the router ids and the activation image.
// Everything about WHERE experts come from, which are resident, the Launch (GPU hits) -> CPU misses -> Combine
// order and the cache budget/admission is the SHARED EXPERT SERVICE's: `ExpertSource` (`expert_source.cpp`),
// `ExpertCache` and `expert_native_*` in `expert_source.cpp`.  This file never launches an expert kernel,
// never picks a slot and never fills one.
#include "strata/core/qwen35_bridge.hpp"
#include "strata/core/qwen35_prefill.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {
namespace {

/// The graph boundary's per-layer state: the scheduler events, the two streams, and the BOUNDED pinned staging
/// (router ids, the activation image, the miss rows).  No expert buffers, no cache - those are the service's.
struct LayerState {
    int layer = -1;
    bool boundary_init = false;
    ggml_backend_t backend = nullptr;
    ggml_backend_event_t pre = nullptr, post = nullptr;
    /// The overlap prototype's join: recorded by the routed seam on the backend stream so that it covers the
    /// shared-expert submission that sits between the pre marker and the routed node.
    ggml_backend_event_t join = nullptr;
    bool pre_fresh = false;   ///< the pre marker recorded `pre` for this layer this round; consumed once
    cudaStream_t work = nullptr, copy = nullptr;
    cudaEvent_t in_ids = nullptr, in_act = nullptr, miss_up = nullptr;
    int32_t* host_ids = nullptr;
    float* host_act = nullptr;
    float* host_miss = nullptr;
    std::vector<int32_t> local_ids;
    std::vector<uint8_t> is_hit;
};

struct State {
    bool initialized = false;
    bool prefill_gpu = false;   ///< the prompt layers run the shared service's GPU expert pipeline
    bool shared_overlap = false;   ///< STRATA_QWEN35_SHARED_OVERLAP=1: decode layers get the pre marker
    std::atomic<bool> seam_armed{false};
    Qwen35BridgeConfig cfg;
    int64_t N = 0, NE = 0, H = 0, FF = 0, K = 0;
    std::unique_ptr<kernels::cpu::ExpertPool> pool;
    std::unique_ptr<FileExpertSource> src;
    ExpertDispatch dispatch;
    std::vector<std::unique_ptr<LayerState>> layers;
    std::atomic<uint64_t> jobs{0}, quants{0}, seconds_us{0};
    uint64_t decode_layers = 0, prefill_layers = 0, prefill_entries = 0;
    uint64_t overlap_marker_layers = 0, overlap_join_layers = 0;
    uint64_t start_hits = 0, start_misses = 0;
    std::mutex routing_mutex;
    std::ofstream routing;
    bool routing_tried = false;
    // routing-detail metadata (default off; STRATA_QWEN35_ROUTING_DETAIL_LOG as a path; request-scoped append)
    std::ofstream routing_detail;
    bool routing_detail_tried = false;
    bool routing_detail_armed = false;
    uint64_t routing_detail_rows = 0;
    uint32_t routing_detail_request = 0;
    // step-7/layer-0 probe (default off, one capture per request; armed after finish_prompt): round 7 of the
    // post-prompt layer-0 T1 calls is the evaluation that produces output index 7.
    bool step_armed = false;
    uint64_t step_base = 0;   ///< S.decode_layers right after finish_prompt (the fixed case: 1)
    int step_round = 0;       ///< layer-0 T1 calls since arming (prompt-tail T1 is not counted: armed later)
    bool step_done = false;
    std::vector<uint8_t> step_a, step_b, step_w;
    std::vector<int32_t> step_ids;
    std::vector<uint8_t> step_hit;
};

State* g_state = nullptr;

int layer_from_name(const char* name) {
    const char* dash = std::strrchr(name, '-');
    return dash != nullptr ? std::atoi(dash + 1) : -1;
}

double now_seconds() {
    static const auto epoch = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

/// The decode shared-expert overlap prototype is opt-in by VALUE: only the exact string "1" enables it.  An
/// unset variable or any other value ("0" included) keeps the existing graph and event order.
bool shared_overlap_requested() {
    const char* v = std::getenv("STRATA_QWEN35_SHARED_OVERLAP");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

/// Step-7/layer-0 probe (diagnostic, default off): exact value "1" enables it.  When unset, every probe branch
/// is a cached-bool test only - no allocation, no clock, no DMA, no sync and no file I/O on the default path.
bool step_probe_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STEP_PROBE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// FNV-1a 64 with the explicit seed below (the same custom seed as the route fingerprint); printed in the record
/// line so the byte count and the seed are part of the evidence.
constexpr uint64_t kStepProbeFnvSeed = 1469598103934665603ull;
uint64_t step_probe_fnv(uint64_t h, const void* p, size_t n) {
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= kPrime; }
    return h;
}

/// Step probe phase 1 (called after the graph pre/work waits, BEFORE the original decode service): round
/// bookkeeping for layer-0 T1 calls and, at round 7 only, the bounded A (MoE input) / W (router weights) D2H on
/// the work stream.  The input is captured before the service runs (no alias with the output).  Shape/type are
/// validated against the real tensors; a mismatch or a CUDA error fails loudly through the original contract.
bool step_probe_begin(State& S, LayerState& L, const ggml_tensor* node, int layer, bool& active) {
    active = false;
    if (!step_probe_requested()) return true;
    if (!S.step_armed || S.step_done || layer != 0 || node->ne[1] != 1) return true;
    ++S.step_round;   // the prompt-tail T1 call happened before arming, so round 7 == output index 7
    if (S.step_round != 7) return true;
    // Guard: the bounded contiguous copies below require complete F32, contiguous single-token A/W and a
    // contiguous single-token F32 output.  The null case is checked FIRST and the rejection message never
    // dereferences a missing source; any other layout is rejected too (the expert computation, streams and
    // scratch are untouched - the probe just does not run).
    const bool shapes_ok =
        node->src[0] != nullptr && node->src[1] != nullptr && node->src[2] != nullptr && node->data != nullptr &&
        node->type == GGML_TYPE_F32 && node->ne[0] == S.H && ggml_nelements(node) == S.H &&
        ggml_is_contiguous(node) && node->src[0]->data != nullptr && node->src[0]->type == GGML_TYPE_F32 &&
        node->src[0]->ne[0] == S.H && ggml_nelements(node->src[0]) == S.H && ggml_is_contiguous(node->src[0]) &&
        node->src[2]->data != nullptr && node->src[2]->type == GGML_TYPE_F32 &&
        ggml_nelements(node->src[2]) == S.K && ggml_is_contiguous(node->src[2]);
    if (!shapes_ok) {
        std::fprintf(stderr, "strata qwen35: step probe: the layer-0 tensors are not the expected contiguous "
                             "F32 single-token layout; probe rejected\n");
        return false;
    }
    const size_t a_bytes = (size_t) S.H * sizeof(float);
    const size_t w_bytes = (size_t) S.K * sizeof(float);
    S.step_a.assign(a_bytes, 0);
    S.step_b.assign(a_bytes, 0);
    S.step_w.assign(w_bytes, 0);
    if (cudaMemcpyAsync(S.step_a.data(), node->src[0]->data, a_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess ||
        cudaMemcpyAsync(S.step_w.data(), node->src[2]->data, w_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess) {
        std::fprintf(stderr, "strata qwen35: step probe: the input/weight D2H could not be enqueued\n");
        return false;
    }
    active = true;
    return true;
}

/// Step probe phase 2 (after the original service returned successfully): B (MoE output) D2H on the same work
/// stream with ONE checked sync (before the bridge's post record), then the existing host_ids/is_hit copies and
/// the bounded raw files + record line.  Any CUDA/file failure fails loudly; nothing prints a hash on failure.
bool step_probe_end(State& S, LayerState& L, const ggml_tensor* node, bool active) {
    if (!active) return true;
    const size_t a_bytes = (size_t) S.H * sizeof(float);
    const size_t w_bytes = (size_t) S.K * sizeof(float);
    if (cudaMemcpyAsync(S.step_b.data(), node->data, a_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess ||
        cudaStreamSynchronize(L.work) != cudaSuccess) {
        std::fprintf(stderr, "strata qwen35: step probe: the output D2H/sync failed\n");
        return false;
    }
    // Guard: every captured float must be finite BEFORE anything is written or marked successful; a NaN/Inf is
    // a real diagnostic failure (no clipping, no tolerance), and nothing is trimmed into a passing result.
    int64_t fa = 0, fb = 0, fw = 0;
    for (int64_t i = 0; i < S.H; ++i) {
        float va, vb;
        std::memcpy(&va, S.step_a.data() + (size_t) i * sizeof(float), sizeof(va));
        std::memcpy(&vb, S.step_b.data() + (size_t) i * sizeof(float), sizeof(vb));
        if (std::isfinite(va)) ++fa;
        if (std::isfinite(vb)) ++fb;
    }
    for (int64_t i = 0; i < S.K; ++i) {
        float vw;
        std::memcpy(&vw, S.step_w.data() + (size_t) i * sizeof(float), sizeof(vw));
        if (std::isfinite(vw)) ++fw;
    }
    if (fa != S.H || fb != S.H || fw != S.K) {
        std::fprintf(stderr, "strata qwen35: step probe: non-finite values in the captured tensors (finite A "
                             "%lld/%lld B %lld/%lld W %lld/%lld); probe failed before any raw/success record\n",
                     (long long) fa, (long long) S.H, (long long) fb, (long long) S.H, (long long) fw,
                     (long long) S.K);
        return false;
    }
    S.step_ids.assign((size_t) S.K, 0);
    S.step_hit.assign((size_t) S.K, 0);
    for (int64_t k = 0; k < S.K; ++k) {
        S.step_ids[(size_t) k] = L.host_ids[k];
        S.step_hit[(size_t) k] = L.is_hit[(size_t) k];
    }
    const char* dir = std::getenv("STRATA_QWEN35_STEP_PROBE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        std::fprintf(stderr, "strata qwen35: step probe: no output directory (STRATA_QWEN35_STEP_PROBE_DIR)\n");
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "strata qwen35: step probe: cannot create %s (%s)\n", dir, ec.message().c_str());
        return false;
    }
    const size_t ids_bytes = (size_t) S.K * sizeof(int32_t);
    struct Part { const char* name; const void* p; size_t n; };
    const Part parts[5] = {{"A.bin", S.step_a.data(), a_bytes}, {"B.bin", S.step_b.data(), a_bytes},
                           {"W.bin", S.step_w.data(), w_bytes}, {"ids.bin", S.step_ids.data(), ids_bytes},
                           {"hit.bin", S.step_hit.data(), (size_t) S.K}};
    for (const Part& pt : parts) {
        const std::string path = std::string(dir) + "/" + pt.name;
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) pt.p, (std::streamsize) pt.n);
        f.flush();
        if (!f.good()) {
            std::fprintf(stderr, "strata qwen35: step probe: cannot write %s\n", path.c_str());
            return false;
        }
        f.close();
        if (f.fail()) {
            std::fprintf(stderr, "strata qwen35: step probe: close failed for %s\n", path.c_str());
            return false;
        }
    }
    std::fprintf(stderr, "strata qwen35: step probe (STRATA_QWEN35_STEP_PROBE=1): round %d index %d layer 0 base "
                         "%llu count %llu H %lld K %lld bytes A %zu B %zu W %zu ids %zu hit %zu finite A %lld B "
                         "%lld W %lld fnv_seed 0x%016llx A 0x%016llx B 0x%016llx W 0x%016llx\n",
                 S.step_round, S.step_round, (unsigned long long) S.step_base,
                 (unsigned long long) S.decode_layers, (long long) S.H, (long long) S.K, a_bytes, a_bytes,
                 w_bytes, ids_bytes, (size_t) S.K, (long long) fa, (long long) fb, (long long) fw,
                 (unsigned long long) kStepProbeFnvSeed,
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_a.data(), a_bytes),
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_b.data(), a_bytes),
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_w.data(), w_bytes));
    S.step_done = true;
    return true;
}

void append_routing(const std::string& payload) {
    State& S = *g_state;
    if (payload.empty()) return;
    std::lock_guard<std::mutex> guard(S.routing_mutex);
    if (!S.routing_tried) {
        S.routing_tried = true;
        S.routing.open(S.cfg.routing_log, std::ios::app | std::ios::binary);
    }
    if (S.routing.is_open()) S.routing << payload;
}

/// The routing-detail diagnostic (default off): STRATA_QWEN35_ROUTING_DETAIL_LOG is a file path and only a
/// non-empty value enables it.  Read once (cached); unset -> every branch below is a cached-pointer test.
const char* routing_detail_path() {
    static const char* v = [] {
        const char* e = std::getenv("STRATA_QWEN35_ROUTING_DETAIL_LOG");
        return (e != nullptr && e[0] != '\0') ? e : nullptr;
    }();
    return v;
}

uint64_t routing_detail_fnv(uint64_t h, const void* p, size_t n) {
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= kPrime; }
    return h;
}

/// Opens the detail file on first use (APPEND: an existing file is never truncated), then writes one request
/// marker and the bounded host residency summary.  False with an explicit message on any open/write/flush
/// failure - the diagnostic was requested, so an incomplete file must not look like a pass.  This runs after
/// the prompt-tail T1 call is already counted and the prompt loan was returned, so the marker is the
/// post-prompt baseline: decode_base is S.decode_layers there (1 in the fixed case), hits_base/misses_base are
/// the per-expert counters there, and only LATER layers are logged; the row sums therefore correspond to the
/// stats DELTAS (rows = (decode_layers - decode_base) * K, hit_sum = decode_hits - hits_base).  map_fnv is a
/// standard FNV-1a64 (offset basis 14695981039346656037, prime 1099511628211) over the resident
/// (layer, expert, slot) int32 tuples in ascending (layer, expert) order, hashed as their native little-endian
/// bytes; the resident count and decode_base are separate marker fields and are NOT part of the digest.  It is
/// read-only host state (no cache/GPU mutation) and only a summary: hashes can collide in principle and the
/// slot payload bytes are not observed, so equal digests do NOT prove equal cache data.  The request ordinal is
/// State-local (it starts at 1 in every process): a path reused across processes is still segmented by its
/// markers, but the N values are not globally increasing.
bool routing_detail_begin(State& S) {
    const char* path = routing_detail_path();
    if (path == nullptr) return true;
    if (!S.routing_detail_tried) {
        S.routing_detail_tried = true;
        S.routing_detail.open(path, std::ios::app | std::ios::binary);
        if (!S.routing_detail.is_open()) {
            std::fprintf(stderr, "strata qwen35: routing detail log cannot open %s\n", path);
            return false;
        }
    }
    ++S.routing_detail_request;
    S.routing_detail_rows = 0;
    S.routing_detail_armed = true;
    uint64_t h = 14695981039346656037ull;
    int64_t resident = 0;
    for (int64_t l = 0; l < S.N; ++l) {
        for (int64_t e = 0; e < S.NE; ++e) {
            const int32_t slot = S.dispatch.cache != nullptr ? S.dispatch.cache->slot_of(l, e) : (int32_t) -1;
            if (slot < 0) continue;
            ++resident;
            const int32_t v[3] = {(int32_t) l, (int32_t) e, slot};
            h = routing_detail_fnv(h, v, sizeof(v));
        }
    }
    S.routing_detail << "#request " << S.routing_detail_request << " decode_base " << S.decode_layers
                     << " hits_base " << S.dispatch.native_hits << " misses_base " << S.dispatch.native_misses
                     << " resident " << resident << " map_fnv 0x" << std::hex << h << std::dec
                     << " layers " << S.N << " experts " << S.NE
                     << " schema before_count,layer,rank,id,hit,slot_after\n";
    S.routing_detail.flush();
    if (!S.routing_detail.good()) {
        std::fprintf(stderr, "strata qwen35: routing detail log write/flush failed (request marker)\n");
        return false;
    }
    return true;
}

LayerState& layer_state(State& S, int layer) {
    if (S.layers[(size_t) layer] == nullptr) {
        S.layers[(size_t) layer] = std::make_unique<LayerState>();
        S.layers[(size_t) layer]->layer = layer;
    }
    return *S.layers[(size_t) layer];
}

/// Releases whatever this layer's boundary created - a PARTIAL failure included - and nulls the handles, so a
/// retry starts clean and shutdown never leaks the pieces a mid-way failure left behind.
void boundary_free(LayerState& L) {
    if (L.work) cudaStreamSynchronize(L.work);
    if (L.copy) cudaStreamSynchronize(L.copy);
    if (L.host_ids) { cudaFreeHost(L.host_ids); L.host_ids = nullptr; }
    L.host_act = nullptr;   // views into the pinned block above
    L.host_miss = nullptr;
    if (L.work) { cudaStreamDestroy(L.work); L.work = nullptr; }
    if (L.copy) { cudaStreamDestroy(L.copy); L.copy = nullptr; }
    if (L.in_ids) { cudaEventDestroy(L.in_ids); L.in_ids = nullptr; }
    if (L.in_act) { cudaEventDestroy(L.in_act); L.in_act = nullptr; }
    if (L.miss_up) { cudaEventDestroy(L.miss_up); L.miss_up = nullptr; }
    if (L.pre) { ggml_backend_event_free(L.pre); L.pre = nullptr; }
    if (L.post) { ggml_backend_event_free(L.post); L.post = nullptr; }
    if (L.join) { ggml_backend_event_free(L.join); L.join = nullptr; }
    L.pre_fresh = false;
    L.local_ids.clear();
    L.is_hit.clear();
    L.boundary_init = false;
}

/// The graph boundary's own buffers: two non-blocking streams, the scheduler events, and one bounded pinned
/// staging block (k ids | n_embd activation | k x n_embd miss rows).
bool boundary_init(LayerState& L, ggml_backend_t backend, int64_t K, int64_t H) {
    boundary_free(L);   // a retry starts clean even after a partial failure
    L.backend = backend;
    bool ok = true;
    // The overlap prototype's join is allocated ONLY when the prototype is latched: the default path
    // allocates exactly the events it did before, and the join consumer cannot run without the latch.
    const bool want_join = g_state != nullptr && g_state->shared_overlap;
    L.pre = ggml_backend_event_new(ggml_backend_get_device(backend));
    L.post = ggml_backend_event_new(ggml_backend_get_device(backend));
    if (want_join) L.join = ggml_backend_event_new(ggml_backend_get_device(backend));
    if (L.pre == nullptr || L.post == nullptr || (want_join && L.join == nullptr)) ok = false;
    if (ok && cudaStreamCreateWithFlags(&L.work, cudaStreamNonBlocking) != cudaSuccess) ok = false;
    if (ok && cudaStreamCreateWithFlags(&L.copy, cudaStreamNonBlocking) != cudaSuccess) ok = false;
    for (cudaEvent_t* e : {&L.in_ids, &L.in_act, &L.miss_up}) {
        if (!ok) break;
        if (cudaEventCreateWithFlags(e, cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    const size_t act = (size_t) H * sizeof(float);
    const size_t off_act = ((size_t) K * sizeof(int32_t) + 63) & ~size_t(63);
    const size_t off_miss = (off_act + act + 63) & ~size_t(63);
    const size_t bytes = off_miss + (size_t) K * act;
    uint8_t* pinned = nullptr;
    if (ok && cudaHostAlloc((void**) &pinned, bytes, cudaHostAllocDefault) != cudaSuccess) ok = false;
    if (!ok) {
        if (pinned != nullptr) cudaFreeHost(pinned);
        boundary_free(L);
        return false;
    }
    L.host_ids = (int32_t*) pinned;
    L.host_act = (float*) (pinned + off_act);
    L.host_miss = (float*) (pinned + off_miss);
    L.is_hit.assign((size_t) K, 0);
    L.boundary_init = true;
    return true;
}

/// The CPU fallback if the external callback did not run: a silent wrong answer is worse.
void seam_cpu_abort(ggml_tensor*, int, int, void*) {
    std::fprintf(stderr, "strata qwen35: a decode MoE node reached the CPU path - the external scheduler seam "
                         "was not installed\n");
    std::fflush(stderr);
    std::abort();
}

void prefill_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    LayerState* L = (LayerState*) userdata;
    State& S = *g_state;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "qwen35 trace: prefill_compute layer %d ith %d nth %d\n", L->layer, ith, nth);
    if (ith != 0 || nth != 1 || L == nullptr) {
        std::fprintf(stderr, "strata qwen35: the prefill expert node must run as one task\n");
        std::abort();
    }
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* ids = dst->src[1];
    const ggml_tensor* w = dst->src[2];
    const int64_t n_tok = x->ne[1];
    // The graph's tensors are strided views (selected_experts keeps nb[1] = n_expert*4 from argsort_top_k),
    // so gather a contiguous host image of the three inputs first.
    static thread_local std::vector<int32_t> idv;
    static thread_local std::vector<float> xv, wv;
    idv.resize((size_t) (n_tok * S.K));
    xv.resize((size_t) (n_tok * S.H));
    wv.resize((size_t) (n_tok * S.K));
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t j = 0; j < S.K; ++j) {
            idv[(size_t) (t * S.K + j)] =
                *(const int32_t*) ((const uint8_t*) ids->data + (size_t) t * ids->nb[1] + (size_t) j * ids->nb[0]);
            wv[(size_t) (t * S.K + j)] =
                *(const float*) ((const uint8_t*) w->data + (size_t) t * w->nb[2] + (size_t) j * w->nb[1]);
        }
        std::memcpy(xv.data() + (size_t) t * S.H, (const uint8_t*) x->data + (size_t) t * x->nb[1],
                    (size_t) S.H * sizeof(float));
    }
    const int32_t* idf = idv.data();
    const float* xf = xv.data();
    const float* wf = wv.data();
    float* out = (float*) dst->data;
    const size_t need = (size_t) n_tok * (size_t) S.K * (size_t) S.H;
    static thread_local std::vector<float> parts;
    if (parts.size() < need) parts.resize(need);
    const double t0 = now_seconds();
    if (trace) {
        std::fprintf(stderr, "qwen35 trace: layer %d dispatch begin n_tok %lld\n", L->layer, (long long) n_tok);
        std::fprintf(stderr, "qwen35 trace: ids ne [%lld %lld] nb [%lld %lld] first:", (long long) ids->ne[0],
                     (long long) ids->ne[1], (long long) ids->nb[0], (long long) ids->nb[1]);
        for (int i = 0; i < 16 && i < (int) (n_tok * S.K); ++i) std::fprintf(stderr, " %d", idf[i]);
        std::fprintf(stderr, " | x nb0 nb1 %lld %lld\n", (long long) x->nb[0], (long long) x->nb[1]);
    }
    S.dispatch.layers = L->layer;
    expert_native_qwen_dispatch(S.dispatch, xf, idf, n_tok, S.K, nullptr, parts.data());
    if (trace) std::fprintf(stderr, "qwen35 trace: layer %d dispatch end (failed %d)\n", L->layer, S.dispatch.failed ? 1 : 0);
    if (S.dispatch.failed) {
        std::fprintf(stderr, "strata qwen35: prefill expert dispatch failed at layer %lld: %s\n",
                     (long long) L->layer, S.dispatch.fail != nullptr ? S.dispatch.fail : "?");
        std::fflush(stderr);
        std::abort();
    }
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t r = 0; r < S.H; ++r) {
            float acc = 0.0f;
            for (int64_t j = 0; j < S.K; ++j)
                acc += wf[(size_t) (t * S.K + j)] * parts[(size_t) ((t * S.K + j) * S.H + r)];
            out[(size_t) (t * S.H + r)] = acc;
        }
    ++S.prefill_layers;
    S.prefill_entries += (uint64_t) (n_tok * S.K);
    S.jobs.fetch_add((uint64_t) (n_tok * S.K), std::memory_order_relaxed);
    S.quants.fetch_add((uint64_t) n_tok, std::memory_order_relaxed);
    S.seconds_us.fetch_add((uint64_t) ((now_seconds() - t0) * 1e6), std::memory_order_relaxed);
}

/// The overlap prototype's decode pre marker.  It computes nothing: it records `pre` on the backend stream at
/// a graph position after the router/activation producers and before the shared expert, so the routed seam
/// that follows can let the CPU miss window overlap the shared expert while still ordering its own hits and
/// combine behind the shared expert's arena reads (through `join`).
bool seam_marker_exec(ggml_backend_t backend, ggml_tensor* node) {
    State& S = *g_state;
    if (!S.shared_overlap) return false;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    const int layer = layer_from_name(node->name);
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_marker layer %d begin\n", layer);
    if (layer < 0 || layer >= (int) S.layers.size() || S.layers[(size_t) layer] == nullptr) return false;
    LayerState& L = *S.layers[(size_t) layer];
    if (!L.boundary_init && !boundary_init(L, backend, S.K, S.H)) {
        std::fprintf(stderr, "strata qwen35: layer %d's graph boundary could not initialise (pre marker)\n", layer);
        return false;
    }
    ggml_backend_event_record(L.pre, backend);
    L.pre_fresh = true;
    ++S.overlap_marker_layers;
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_marker layer %d pre recorded\n", layer);
    return true;
}

bool seam_exec(ggml_backend_t backend, ggml_tensor* node) {
    State& S = *g_state;
    ExpertDispatch& d = S.dispatch;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    const int layer = layer_from_name(node->name);
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_exec layer %d begin\n", layer);
    if (layer < 0 || layer >= (int) S.layers.size() || S.layers[(size_t) layer] == nullptr) return false;
    LayerState& L = *S.layers[(size_t) layer];
    if (!L.boundary_init && !boundary_init(L, backend, S.K, S.H)) {
        std::fprintf(stderr, "strata qwen35: layer %d's graph boundary could not initialise\n", layer);
        return false;
    }
    const int64_t K = S.K;
    const double t0 = now_seconds();
    // The overlap prototype: the decode pre marker already recorded `pre` before the shared expert was
    // submitted.  `join`, recorded now, covers the whole backend-stream prefix (router + shared expert); the
    // work stream waits on it, so the CPU miss window stays free to overlap the shared expert while the hits
    // and the combine cannot touch arena regions the shared expert may still be reading.  Without a fresh
    // marker pre (the default path) this is the ordinary pre record.
    const bool fresh_pre = node->ne[1] == 1 && L.pre_fresh && L.join != nullptr;
    L.pre_fresh = false;
    if (fresh_pre) {
        ggml_backend_event_record(L.join, backend);
        cudaStreamWaitEvent(L.work, (cudaEvent_t) L.join->context, 0);
        ++S.overlap_join_layers;
        if (trace) std::fprintf(stderr, "qwen35 trace: seam layer %d join recorded (fresh pre)\n", layer);
    } else {
        ggml_backend_event_record(L.pre, backend);
    }
    cudaStreamWaitEvent(L.copy, (cudaEvent_t) L.pre->context, 0);
    cudaStreamWaitEvent(L.work, (cudaEvent_t) L.pre->context, 0);
    // Step probe (default off): the graph pre/work dependencies are established; capture A/W BEFORE the
    // original service runs so no output/alias can affect the input snapshot.
    bool step_active = false;
    if (!step_probe_begin(S, L, node, layer, step_active)) return false;
    std::string err;
    if (node->ne[1] == 1) {
        // The WHOLE decode layer's routed-MoE dispatch (ids transfer, residency, GPU hits, CPU misses, upload,
        // combine, admission) is the shared service's; the boundary only orders it on its own streams.
        core::NativeDecodeStage st;
        st.work_stream = L.work;
        st.copy_stream = L.copy;
        st.host_ids = L.host_ids;
        st.host_act = L.host_act;
        st.host_miss = L.host_miss;
        st.is_hit = L.is_hit.data();
        if (!expert_native_decode_layer(d, layer, K, node->src[0]->data, node->src[1]->data, node->src[2]->data,
                                        node->data, st, err)) {
            std::fprintf(stderr, "strata qwen35: layer %d: %s (%s)\n", layer, err.c_str(),
                         d.fail != nullptr ? d.fail : "no dispatch message");
            return false;
        }
        // Step probe (default off): capture B on the same work stream after the service, with one checked sync;
        // `count` in its record line is S.decode_layers BEFORE this layer's increment (241 before index 7).
        if (!step_probe_end(S, L, node, step_active)) return false;
        if (trace) std::fprintf(stderr, "qwen35 trace: seam layer %d ids [%d %d %d %d %d %d %d %d]\n", layer,
                                L.host_ids[0], L.host_ids[1], L.host_ids[2], L.host_ids[3], L.host_ids[4],
                                L.host_ids[5], L.host_ids[6], L.host_ids[7]);
        if (!S.cfg.routing_log.empty()) {
            std::string routing;
            for (int64_t r = 0; r < K; ++r) {
                routing += std::to_string(layer);
                routing += ',';
                routing += std::to_string(L.host_ids[r]);
                routing += '\n';
            }
            append_routing(routing);
        }
        // Routing-detail rows (default off; request-scoped, hard-capped).  Schema (see the request marker):
        // before_count,layer,rank,id,hit,slot_after.  before_count is S.decode_layers BEFORE this layer's
        // increment; the marker's decode_base is the already-counted prompt-tail T1 call, so the logged rows
        // start after it and cover only post-prompt layers (the fixed short teacher8 run: 2240 rows / 280
        // layers, before_count 1..280).  2248 is only a hard cap: for a longer run the log is a prefix and the
        // row sums then match the stats deltas only up to that prefix - they never cover the whole request.
        // slot_after is the read-only residence query at service return (admissions made inside the call are
        // already visible); it is NOT the slot the kernel actually used.  With the diagnostic enabled, each
        // layer's rows are flushed and the stream is checked, so a delayed write error stops the run through
        // the seam error path; that per-layer flush perturbs scheduling - not a speed-comparison configuration.
        if (S.routing_detail_armed && S.routing_detail_rows < 2248) {
            const int64_t before_count = (int64_t) S.decode_layers;
            for (int64_t r = 0; r < K && S.routing_detail_rows < 2248; ++r) {
                const int32_t id = L.host_ids[r];
                const int32_t slot_after = (S.dispatch.cache != nullptr && id >= 0 && id < S.NE)
                                               ? S.dispatch.cache->slot_of(layer, id)
                                               : (int32_t) -1;
                S.routing_detail << before_count << ',' << layer << ',' << r << ',' << id << ','
                                 << (int) (L.is_hit[(size_t) r] != 0) << ',' << slot_after << '\n';
                ++S.routing_detail_rows;
            }
            S.routing_detail.flush();
            if (!S.routing_detail.good()) {
                // The diagnostic was requested: an incomplete file must not look like a pass.
                std::fprintf(stderr, "strata qwen35: routing detail log write/flush failed at layer %d\n", layer);
                return false;
            }
        }
        ++S.decode_layers;
        S.jobs.fetch_add((uint64_t) K, std::memory_order_relaxed);
    } else {
        // Prompt layer: the shared service's GPU expert pipeline (grouped experts, resident cache slots, the
        // bounded cold ring, the MMQ products).  Its counters carry the evidence.
        const int64_t T = node->ne[1];
        if (!qwen_prefill_experts(d, layer, T, node->src[0]->data, node->src[0]->nb[1], node->src[1]->data,
                                  node->src[1]->nb[1], node->src[2]->data, node->src[2]->nb[1], node->src[2]->nb[2],
                                  node->data, L.work, err)) {
            std::fprintf(stderr, "strata qwen35: prompt layer %d: %s (%s)\n", layer, err.c_str(),
                         d.fail != nullptr ? d.fail : "no dispatch message");
            return false;
        }
        S.prefill_layers += 1;
        S.prefill_entries += (uint64_t) (T * K);
        S.jobs.fetch_add((uint64_t) (T * K), std::memory_order_relaxed);
    }
    if (cudaEventRecord((cudaEvent_t) L.post->context, L.work) != cudaSuccess) return false;
    ggml_backend_event_wait(backend, L.post);
    {
        const cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: decode seam CUDA error: %s\n", cudaGetErrorString(cerr));
            return false;
        }
    }
    S.seconds_us.fetch_add((uint64_t) ((now_seconds() - t0) * 1e6), std::memory_order_relaxed);
    return true;
}

bool seam_callback(ggml_backend_t backend, ggml_tensor* node, bool ask, void*) {
    if (g_state == nullptr || !g_state->initialized || !g_state->seam_armed.load(std::memory_order_relaxed))
        return false;
    if (node->op != GGML_OP_CUSTOM) return false;
    if (std::strncmp(node->name, "strata_moe_pre-", 15) == 0) {
        // The overlap prototype's decode pre marker: exactly one device source (the activation, one token).
        // Any other node that happens to carry the name is not claimed.
        if (node->src[0] == nullptr || node->src[1] != nullptr || node->src[0]->type != GGML_TYPE_F32 ||
            node->src[0]->ne[0] != g_state->H || node->src[0]->ne[1] != 1)
            return false;
        if (ask) return true;
        return seam_marker_exec(backend, node);
    }
    if (std::strncmp(node->name, "ffn_moe_out-", 12) != 0) return false;
    // llama.cpp's `cb()` renames BOTH custom MoE nodes to ffn_moe_out-<layer>; ours carry exactly the three
    // device inputs (x, ids, weights).  Any other custom node (the CPU fallback has 6 inputs) is not claimed.
    if (node->src[0] == nullptr || node->src[1] == nullptr || node->src[2] == nullptr || node->src[3] != nullptr)
        return false;
    if (ask) return true;
    return seam_exec(backend, node);
}

}  // namespace

bool qwen35_bridge_init(const Qwen35BridgeConfig& cfg, std::string& err) {
    if (g_state != nullptr) { err = "qwen35: the bridge is already initialised"; return false; }
    if (cfg.model.empty() || cfg.pack_dir.empty()) { err = "qwen35: --model and the pack directory are required"; return false; }
    if (cfg.n_layers <= 0 || cfg.n_expert <= 0 || cfg.n_embd <= 0 || cfg.n_ff <= 0 || cfg.k <= 0) {
        err = "qwen35: the model geometry is empty";
        return false;
    }
    if (cfg.k > 16) { err = "qwen35: k > 16 is not supported by the dispatch buffers"; return false; }
    if (!kernels::cpu::expert_layout_load(cfg.pack_dir, cfg.n_layers, cfg.n_expert, cfg.n_embd, cfg.n_ff, err))
        return false;
    const auto& lay = kernels::cpu::expert_layout();
    if (!lay.native) { err = "qwen35: the pack has no native_experts.txt"; return false; }
    for (int64_t l = 0; l < (int64_t) lay.fmt.size(); ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
            err = "qwen35: layer " + std::to_string(l) + "'s expert formats have no GPU kernels (ggml types " +
                  std::to_string(f.gu_type) + "/" + std::to_string(f.d_type) + ")";
            return false;
        }
    }
    std::unique_ptr<State> S(new State());
    S->cfg = cfg;
    S->N = cfg.n_layers;
    S->NE = cfg.n_expert;
    S->H = cfg.n_embd;
    S->FF = cfg.n_ff;
    S->K = cfg.k;
    S->shared_overlap = shared_overlap_requested();
    S->layers.resize((size_t) S->N);
    S->src = std::make_unique<FileExpertSource>();
    S->src->set_gguf(cfg.model);
    if (!S->src->open(cfg.pack_dir, S->N, S->NE, err)) return false;
    S->pool = std::make_unique<kernels::cpu::ExpertPool>(cfg.pool_workers, true, true);
    S->dispatch.pool = S->pool.get();
    S->dispatch.src = S->src.get();
    S->dispatch.n_expert = S->NE;
    S->dispatch.n_embd = S->H;
    S->dispatch.n_ff = S->FF;
    S->dispatch.native_k = cfg.k;
    // The prompt path runs on the GPU when the MMQ build covers every layer's formats and the tiles fit.
    {
        namespace mmq = strata::prefill::mmq;
        S->prefill_gpu = qwen_prefill_mmq_available();
        for (int64_t l = 0; S->prefill_gpu && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            S->prefill_gpu = mmq::supported(f.gu_type) && mmq::supported(f.d_type) &&
                             mmq::fits(f.gu_type, 2 * f.n_ff) && mmq::fits(f.d_type, f.n_embd);
        }
    }
    if (!expert_native_cache_open(S->dispatch, cfg.cache_slots, cfg.profile, err)) return false;
    g_state = S.release();
    std::fprintf(stderr, "strata qwen35: bridge ready: %lld layers x %lld experts, %lld x %lld experts top-%lld, "
                         "workers %d, native tier %s, prompt experts %s\n", (long long) cfg.n_layers,
                 (long long) cfg.n_expert, (long long) cfg.n_embd, (long long) cfg.n_ff, (long long) cfg.k,
                 g_state->pool->workers(), g_state->dispatch.cache != nullptr ? "on" : "off",
                 g_state->prefill_gpu ? "GPU/MMQ (grouped, resident cache + bounded cold ring)"
                                      : "CPU fallback");
    if (g_state->shared_overlap)
        std::fprintf(stderr, "strata qwen35: decode shared-expert overlap prototype ON (STRATA_QWEN35_SHARED_OVERLAP=1): "
                             "one pre marker per decode layer, combine ordered after the shared expert\n");
    g_state->initialized = true;
    return true;
}

void qwen35_bridge_shutdown() {
    if (g_state == nullptr) return;
    State& S = *g_state;
    S.initialized = false;
    if (S.routing.is_open()) S.routing.close();
    if (S.routing_detail_armed) {
        // Best-effort final flush of the default-off detail log: every layer's rows were already flushed and
        // checked on the write path, so a failure here is reported on stderr only - it cannot change the return
        // code or the already-printed stats summary, and it never aborts the normal resource release below.
        S.routing_detail.flush();
        if (!S.routing_detail.good())
            std::fprintf(stderr, "strata qwen35: routing detail log final flush failed (best effort, rc unchanged)\n");
        else
            std::fprintf(stderr, "strata qwen35: routing detail log: request %u rows %llu (last request, cap 2248)\n",
                         S.routing_detail_request, (unsigned long long) S.routing_detail_rows);
    }
    for (auto& lp : S.layers) {
        if (lp == nullptr) continue;
        boundary_free(*lp);   // partial initializations included: whatever was created is released here
    }
    S.layers.clear();
    qwen_prefill_close(S.dispatch);
    expert_native_close(S.dispatch);
    S.dispatch.pool = nullptr;
    S.dispatch.src = nullptr;
    S.pool.reset();
    S.src->close();
    S.src.reset();
    delete g_state;
    g_state = nullptr;
}

void qwen35_bridge_stats(Qwen35BridgeStats& out) {
    out = Qwen35BridgeStats{};
    if (g_state == nullptr) return;
    State& S = *g_state;
    out.decode_layers = S.decode_layers;
    out.prefill_layers = S.prefill_layers;
    out.decode_hits = (uint64_t) S.dispatch.native_hits;
    out.decode_misses = (uint64_t) S.dispatch.native_misses;
    out.prefill_entries = S.prefill_entries;
    out.jobs = S.jobs.load(std::memory_order_relaxed);
    out.expert_seconds = (double) S.seconds_us.load(std::memory_order_relaxed) / 1e6;
    out.ms_ids = S.dispatch.native_stats.ms_ids;
    out.ms_launch = S.dispatch.native_stats.ms_launch;
    out.ms_misses = S.dispatch.native_stats.ms_misses;
    out.ms_combine = S.dispatch.native_stats.ms_combine;
    out.ms_queue = S.dispatch.native_stats.ms_queue;
    out.hit_overlap = S.dispatch.native_stats.hit_overlap;
    out.hit_late = S.dispatch.native_stats.hit_late;
    out.hit_inflight = S.dispatch.native_stats.hit_inflight;
    out.all_hit = S.dispatch.native_stats.all_hit;
    out.prefill_layers_gpu = S.dispatch.native_stats.prefill_layers_gpu;
    out.prefill_resident = S.dispatch.native_stats.prefill_resident;
    out.prefill_streamed = S.dispatch.native_stats.prefill_streamed;
    out.prefill_product_calls = S.dispatch.native_stats.prefill_product_calls;
    out.prefill_stream_bytes = S.dispatch.native_stats.prefill_stream_bytes;
    out.prefill_workspace_bytes = S.dispatch.native_stats.prefill_workspace_bytes;
    out.prefill_ring_bytes = S.dispatch.native_stats.prefill_ring_bytes;
    out.prefill_loan_slots = S.dispatch.native_stats.prefill_loan_slots;
    out.prefill_loan_bytes = S.dispatch.native_stats.prefill_loan_bytes;
    out.prefill_owned_bytes = S.dispatch.native_stats.prefill_owned_bytes;
    out.prefill_loan_restored = S.dispatch.native_stats.prefill_loan_restored;
    out.prefill_loan_refill_ms = S.dispatch.native_stats.prefill_loan_refill_ms;
    out.prefill_loan_why = S.dispatch.native_stats.prefill_loan_why;
    out.prefill_gpu_ms = S.dispatch.native_stats.prefill_ms;
    out.shared_overlap_marker_layers = S.overlap_marker_layers;
    out.shared_overlap_join_layers = S.overlap_join_layers;
    if (S.dispatch.cache != nullptr) {
        out.cache_slots = (uint64_t) S.dispatch.cache->slots();
        out.cache_bytes = (uint64_t) S.dispatch.cache->bytes();
        out.cache_resident = (uint64_t) S.dispatch.cache->resident();
        // The REAL resident tensor payload (admitted pairs' own blob sizes), distinct from the arena bytes
        // that the slots' max-blob sizing and the never-filled slots inflate.
        const auto& lay = kernels::cpu::expert_layout();
        uint64_t payload = 0;
        for (int64_t l = 0; l < (int64_t) lay.fmt.size() && l < S.N; ++l)
            for (int64_t e = 0; e < S.NE; ++e)
                if (S.dispatch.cache->slot_of(l, e) >= 0) payload += (uint64_t) lay.blob_bytes(l);
        out.cache_payload_bytes = payload;
    }
}

uint64_t qwen35_native_jobs() { return g_state ? g_state->jobs.load(std::memory_order_relaxed) : 0; }
uint64_t qwen35_native_quants() { return g_state ? g_state->quants.load(std::memory_order_relaxed) : 0; }
double qwen35_native_seconds() {
    return g_state ? (double) g_state->seconds_us.load(std::memory_order_relaxed) / 1e6 : 0.0;
}

bool qwen35_native_enabled() { return g_state != nullptr && g_state->initialized; }

bool qwen35_native_layer_enabled(int layer, int64_t n_tokens) {
    (void) layer; (void) n_tokens;
    // Diagnostic A/B: let llama.cpp's own graph compute the experts (the anchor against this bridge).
    if (std::getenv("STRATA_QWEN35_GGML_MOE") != nullptr) return false;
    return qwen35_native_enabled();
}

bool qwen35_shared_overlap_tensor(ggml_tensor* x) {
    // EXACTLY the marker condition `qwen35_native_moe` uses (`S.shared_overlap && x->ne[1] == 1`, with the
    // same tensor) plus the A/B bypass: the llama-side anchor fires iff the routed node was left unexpanded.
    if (g_state == nullptr || !g_state->initialized || !g_state->shared_overlap) return false;
    if (std::getenv("STRATA_QWEN35_GGML_MOE") != nullptr) return false;
    return x != nullptr && x->ne[1] == 1;
}

void qwen35_install_decode_seam(ggml_backend_sched* sched) {
    if (sched != nullptr && qwen35_native_enabled()) ggml_backend_sched_set_external_callback(sched, seam_callback, nullptr);
}

void qwen35_arm_decode_seam(bool on) {
    if (g_state != nullptr) g_state->seam_armed.store(on, std::memory_order_relaxed);
}

bool qwen35_bridge_finish_prompt() {
    if (g_state == nullptr) return true;
    std::string err;
    if (!qwen_prefill_loan_return(g_state->dispatch, err)) {
        std::fprintf(stderr, "strata qwen35: the prompt loan was not returned: %s\n", err.c_str());
        return false;
    }
    // Step probe (default off): the prompt phase is over; the post-prompt layer-0 T1 calls are counted from
    // here and the 7th produces output index 7.  `base` is the native decode_layers at this boundary (the
    // prompt-tail T1 call included - 1 for the fixed case), so the target's count is base + 6 * layers.
    if (step_probe_requested() && !g_state->step_done) {
        g_state->step_armed = true;
        g_state->step_base = g_state->decode_layers;
        std::fprintf(stderr, "strata qwen35: step probe (STRATA_QWEN35_STEP_PROBE=1): armed after finish_prompt, "
                             "base %llu\n", (unsigned long long) g_state->step_base);
    }
    // Routing-detail diagnostic (default off): the request boundary is here - after the prompt-tail T1 call
    // (already counted in decode_layers) and after the prompt loan return.  The marker records this post-prompt
    // baseline (decode_base, hits_base, misses_base) and only the layers after it are logged, so the row sums
    // are the stats deltas.  A failure is a real error.
    if (!routing_detail_begin(*g_state)) return false;
    return true;
}

void qwen35_report_decode_seam() {
    if (g_state == nullptr) return;
    Qwen35BridgeStats st;
    qwen35_bridge_stats(st);
    std::fprintf(stderr, "strata qwen35: expert service: decode layers %llu (hits %llu, misses %llu), prefill "
                         "layers %llu (%llu entries), jobs %llu, expert time %.3f s, cache %llu slots / %.2f GiB "
                         "(%llu resident)\n",
                 (unsigned long long) st.decode_layers, (unsigned long long) st.decode_hits,
                 (unsigned long long) st.decode_misses, (unsigned long long) st.prefill_layers,
                 (unsigned long long) st.prefill_entries, (unsigned long long) st.jobs, st.expert_seconds,
                 (unsigned long long) st.cache_slots, (double) st.cache_bytes / 1073741824.0,
                 (unsigned long long) st.cache_resident);
    std::fprintf(stderr, "strata qwen35: shared tier phases: ids %.1f ms, launch %.1f ms, misses %.1f ms, combine "
                         "%.1f ms, admit %.1f ms; hit/miss overlap: inflight-at-launch %lld, done-at-miss-end %lld, "
                         "late %lld, all-hit %lld\n",
                 st.ms_ids, st.ms_launch, st.ms_misses, st.ms_combine, st.ms_queue,
                 (long long) st.hit_inflight, (long long) st.hit_overlap, (long long) st.hit_late,
                 (long long) st.all_hit);
    std::fprintf(stderr, "strata qwen35: prompt experts (GPU/MMQ): layers %lld, resident %lld, cold-streamed %lld, "
                         "MMQ product launches %lld, streamed %.1f MiB, workspace %.1f MiB, ring %.1f MiB, %.1f ms\n",
                 (long long) st.prefill_layers_gpu, (long long) st.prefill_resident,
                 (long long) st.prefill_streamed, (long long) st.prefill_product_calls,
                 (double) st.prefill_stream_bytes / 1048576.0, (double) st.prefill_workspace_bytes / 1048576.0,
                 (double) st.prefill_ring_bytes / 1048576.0, st.prefill_gpu_ms);
    if (st.prefill_loan_slots > 0) {
        std::fprintf(stderr, "strata qwen35: prompt workspace loan: %lld cache slots / %.2f MiB borrowed; "
                             "%lld experts refilled in %.1f ms\n", (long long) st.prefill_loan_slots,
                     (double) st.prefill_loan_bytes / 1048576.0, (long long) st.prefill_loan_restored,
                     st.prefill_loan_refill_ms);
    } else if (st.prefill_owned_bytes > 0) {
        std::fprintf(stderr, "strata qwen35: prompt workspace: owned buffers %.2f MiB (%s)\n",
                     (double) st.prefill_owned_bytes / 1048576.0,
                     st.prefill_loan_why.empty() ? "no loan" : st.prefill_loan_why.c_str());
    }
    if (st.shared_overlap_marker_layers > 0 || st.shared_overlap_join_layers > 0) {
        std::fprintf(stderr, "strata qwen35: shared-expert overlap prototype: pre markers %llu, consumed by the "
                             "routed seam %llu (event ordering evidence only, not a speed measurement)\n",
                     (unsigned long long) st.shared_overlap_marker_layers,
                     (unsigned long long) st.shared_overlap_join_layers);
    }
}

ggml_tensor* qwen35_reference_moe_marker(ggml_context*, ggml_cgraph*, ggml_tensor* source, ggml_tensor*, ggml_tensor*,
                                         int, bool) {
    return source;
}

ggml_tensor* qwen35_native_moe(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x, ggml_tensor* ids,
                               ggml_tensor* weights, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, int layer) {
    State* S = g_state;
    if (S == nullptr || !S->initialized || layer < 0 || layer >= (int) S->N) return nullptr;
    // weights is llama.cpp's 3D [1, n_expert_used, n_tokens] (this checkout); its per-token k weights are
    // contiguous, which is what the combine kernel and the host prefill combine index.
    if (!(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32 &&
          x->ne[0] == S->H && ids->ne[0] == S->K && weights->ne[0] == 1 && weights->ne[1] == S->K &&
          ids->ne[1] == x->ne[1] && weights->ne[2] == x->ne[1] && gate != nullptr && up != nullptr &&
          down != nullptr)) {
        std::fprintf(stderr, "strata qwen35: unexpected MoE node shapes at layer %d: x ne [%lld %lld %lld %lld] "
                             "type %d, ids ne [%lld %lld] type %d, weights ne [%lld %lld %lld] type %d (expected "
                             "n_embd %lld, k %lld)\n",
                     layer, (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3],
                     (int) x->type, (long long) ids->ne[0], (long long) ids->ne[1], (int) ids->type,
                     (long long) weights->ne[0], (long long) weights->ne[1], (long long) weights->ne[2],
                     (int) weights->type, (long long) S->H, (long long) S->K);
        std::fflush(stderr);
        std::abort();
    }
    LayerState& L = layer_state(*S, layer);
    char name[64];
    if (std::getenv("STRATA_QWEN35_TRACE") != nullptr)
        std::fprintf(stderr, "qwen35 trace: native_moe layer %d n_tok %lld\n", layer, (long long) x->ne[1]);
    // Opt-in decode overlap prototype: a compute-free pre marker right after the router nodes and before the
    // shared expert (whose nodes the caller anchors next, see qwen35moe.cpp).  The routed node below is then
    // NOT expanded here: the caller's add pulls it in on its DFS after the shared expert, giving the order
    // marker -> shared expert -> routed external -> add.  The marker records `pre`; the routed seam joins it.
    const bool overlap_decode = S->shared_overlap && x->ne[1] == 1;
    if (overlap_decode) {
        ggml_tensor* marker_srcs[1] = {x};
        ggml_tensor* marker = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, marker_srcs, 1, seam_cpu_abort, 1, &L);
        std::snprintf(name, sizeof name, "strata_moe_pre-%d", layer);
        ggml_set_name(marker, name);
        ggml_build_forward_expand(graph, marker);
    }
    if (x->ne[1] == 1 || S->prefill_gpu) {
        // Both shapes carry exactly the three device inputs; the scheduler's external callback executes them:
        // one token through the decode seam, a prompt through the shared service's GPU expert pipeline.
        ggml_tensor* srcs[3] = {x, ids, weights};
        ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) S->H, x->ne[1], 1, 1, srcs, 3,
                                           seam_cpu_abort, 1, &L);
        std::snprintf(name, sizeof name, "ffn_moe_out-%d", layer);
        ggml_set_name(node, name);
        if (!overlap_decode) ggml_build_forward_expand(graph, node);
        return node;
    }
    // CPU fallback (no MMQ build/coverage): a CPU custom op.  gate/up/down are inputs only so the scheduler
    // places it on the CPU backend (the expert bytes come from the shared source, not from llama.cpp's tensors).
    ggml_tensor* srcs[6] = {x, ids, weights, gate, up, down};
    ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) S->H, x->ne[1], 1, 1, srcs, 6, prefill_compute, 1, &L);
    std::snprintf(name, sizeof name, "ffn_moe_cpu-%d", layer);
    ggml_set_name(node, name);
    ggml_build_forward_expand(graph, node);
    return node;
}

}  // namespace strata::core
