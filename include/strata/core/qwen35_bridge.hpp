// include/strata/core/qwen35_bridge.hpp - the Qwen35 model graph's expert boundary (POC).
//
// The Qwen3.6-35B-A3B text graph is llama.cpp's (`LLM_ARCH_QWEN35MOE`: GDN + full attention + shared expert);
// the ROUTED experts are Strata's: this bridge turns the graph's routed-MoE node into a CUSTOM op whose
// dispatcher is the shared expert service (`ExpertSource` + `ExpertCache` + `ExpertPool` +
// `expert_native_qwen_dispatch`).  The llama.cpp checkout is patched (LLAMA_STRATA_QWEN35) to call the
// functions below from `build_moe_ffn` and to install the external-producer callback that executes the decode
// node on the device, so activation and router ids never make an early host round trip.
//
// Two shapes:
//   * decode (1 token): the node's inputs {x, ids, weights} stay device-resident; the scheduler's external
//     callback computes the resident (GPU) experts with `native_expert_grouped`, runs the misses on the CPU
//     pool (which reads the GGUF's three role slices directly), and combines on the device.  All-hit layers
//     transfer only the 8 router ids.
//   * prefill (>1 token): the shared service's GPU/MMQ prompt pipeline (`qwen35_prefill.cu`: experts grouped
//     by expert, resident ExpertCache device slots, a bounded pinned cold ring, the mature MMQ products, and
//     the cache-tail workspace loan/refill) when the MMQ build covers every layer's formats; otherwise the
//     CPU custom-op fallback (the gate/up/down inputs only pin it to the CPU backend, every entry is a miss
//     through the same shared dispatch, combined on the host).
//
// Opt-in decode shared-expert overlap prototype (STRATA_QWEN35_SHARED_OVERLAP=1; one-token graphs only).  The
// bridge inserts a compute-free pre marker CUSTOM node (`strata_moe_pre-<layer>`, source = the activation)
// right after the router nodes; the llama.cpp caller anchors the shared-expert nodes before the routed node,
// so the decode order becomes marker -> shared expert -> routed external -> add.  The marker records `pre`
// early; the routed seam records `join` (covering the shared expert) and makes its work stream wait on it, so
// the CPU miss window can overlap the shared expert while the combine cannot race its arena reads.  Unset or
// any value other than "1" keeps the unmodified order (the routed node is expanded at the hook, as before).
#pragma once

#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_sched;

namespace strata::core {

/// Everything the bridge needs; the runner fills it from the model's own metadata.
struct Qwen35BridgeConfig {
    std::string model;        ///< the GGUF (native_experts.txt's spans are absolute in this file)
    std::string pack_dir;     ///< where native_experts.txt lives
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int64_t k = 0;
    /// GPU expert tier: number of resident experts (0 = cache off).  Slots are sized for the largest layer blob.
    int64_t cache_slots = 0;
    /// Optional expert profile (tools/make_profile.py format) ranked hottest first: fills the slots at startup.
    std::string profile;
    /// Optional diagnostic: one "layer,expert" line per routed (token, rank) entry.
    std::string routing_log;
    int pool_workers = 0;     ///< 0 = every physical core but the first
};

struct Qwen35BridgeStats {
    uint64_t decode_layers = 0;
    uint64_t prefill_layers = 0;
    uint64_t decode_hits = 0, decode_misses = 0;
    uint64_t prefill_entries = 0;
    uint64_t jobs = 0;
    uint64_t cache_slots = 0, cache_bytes = 0, cache_resident = 0;
    /// The REAL resident payload: sum of the admitted pairs' actual blob bytes - not the arena's allocated
    /// bytes, which carry per-slot max-blob padding and slots that were never filled.
    uint64_t cache_payload_bytes = 0;
    double expert_seconds = 0.0;
    /// The shared tier's own phase timers (ms) and the measured hit/miss overlap.
    double ms_ids = 0, ms_launch = 0, ms_misses = 0, ms_combine = 0, ms_queue = 0;
    int64_t hit_overlap = 0, hit_late = 0, hit_inflight = 0, all_hit = 0;
    /// The prompt path's real expert-pipeline counters (grouped experts, resident cache, bounded cold ring, MMQ).
    int64_t prefill_layers_gpu = 0, prefill_resident = 0, prefill_streamed = 0, prefill_product_calls = 0;
    uint64_t prefill_stream_bytes = 0, prefill_workspace_bytes = 0, prefill_ring_bytes = 0;
    /// The prompt workspace's loan: the borrowed cache slots/bytes vs the owned fallback (cache-off).
    int64_t prefill_loan_slots = 0, prefill_loan_restored = 0;
    uint64_t prefill_loan_bytes = 0, prefill_owned_bytes = 0;
    double prefill_loan_refill_ms = 0;
    std::string prefill_loan_why;
    double prefill_gpu_ms = 0;
    /// The decode shared-expert overlap prototype (opt-in, STRATA_QWEN35_SHARED_OVERLAP=1).  `marker_layers`
    /// counts the decode layers whose pre marker recorded the early event; `join_layers` counts the decode
    /// callbacks that consumed one (and so ordered their work stream behind the shared expert).  These prove
    /// the graph/event ordering was installed and used; they are NOT a speed measurement.
    uint64_t shared_overlap_marker_layers = 0;
    uint64_t shared_overlap_join_layers = 0;
};

bool qwen35_bridge_init(const Qwen35BridgeConfig& cfg, std::string& err);
void qwen35_bridge_shutdown();
void qwen35_bridge_stats(Qwen35BridgeStats& out);

/// The prompt -> decode transition: returns the prompt path's loan (refills the lent cache slots and restores
/// their residency) so the decode tier sees them again.  The runner calls it after the prompt loop and MUST
/// NOT decode when it returns false: a lent slot could not be refilled (those slots stay `kNotResident`).
bool qwen35_bridge_finish_prompt();

/// Counters the prototype exposed (kept: the runner's summary prints them).
uint64_t qwen35_native_jobs();
uint64_t qwen35_native_quants();
double qwen35_native_seconds();

/// ---- the llama.cpp graph hooks (called from the patched checkout) ----
bool qwen35_native_enabled();
bool qwen35_native_layer_enabled(int layer, int64_t n_tokens);
/// The overlap prototype's anchor predicate, evaluated on the SAME tensor `qwen35_native_moe` saw: true iff
/// that hook left this layer's routed node unexpanded (the latched prototype and a one-token MoE input).  The
/// llama-side anchor must fire on exactly this condition - a cropped one-token layer (the prompt's last layer
/// included) would otherwise land the shared expert after the routed external node.  The environment is read
/// only here (never re-read in the model graph).
bool qwen35_shared_overlap_tensor(struct ggml_tensor* x);
void qwen35_install_decode_seam(ggml_backend_sched* sched);
/// Arms the decode seam.  The scheduler callback is registered when the context is created (the patched
/// llama-context.cpp), but it stays inert until the runner is past graph reservation: claims made during
/// `ggml_backend_sched_reserve` would put a device producer into the reserve graph, which executes nothing.
void qwen35_arm_decode_seam(bool on);
void qwen35_report_decode_seam();
ggml_tensor* qwen35_reference_moe_marker(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* source,
    ggml_tensor* ids, ggml_tensor* weights, int layer, bool start);
ggml_tensor* qwen35_native_moe(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x,
    ggml_tensor* ids, ggml_tensor* weights, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, int layer);

}  // namespace strata::core
