// include/strata/core/qwen35_prefill.hpp - the Qwen35 prompt layer through Strata's original expert pipeline.
//
// THE SAME SHAPE AS THE FLASH-NEXT PROMPT PATH (src/prefill/prefill.cpp, the MMQ branch), generalised to the
// model descriptor instead of the fixed 2560/640/512 geometry:
//
//   * the layer's routed entries are grouped BY EXPERT;
//   * a resident expert is read from its ExpertCache device slot (`device_slot`);
//   * a cold expert is copied through a BOUNDED pinned host ring onto the copy stream (overlapping the
//     previous group's MMQ products), never pinned whole and never materialised as a host blob;
//   * each group of experts is gathered into the MMQ group buffer (`mmq::gather_native`) and multiplied with
//     the mature MMQ products (`mmq::Product`, the pinned llama.cpp kernels) for gate/up, then SwiGLU, the
//     hidden quantizer, and the down product;
//   * the entries' rows are combined back to the tokens with the device router weights.
//
// The workspace is the cache's LENT TAIL when the run is cache-on: `expert_native_loan_plan` marks the tail
// slots' residents as misses (they stream through the same ring) and hands back the region; the workspace,
// group buffers and device ring are bump-allocated inside it, and `qwen_prefill_loan_return` refills the lent
// experts and restores residency before the decode tier runs.  Cache-off (or a tail too small) falls back to
// the dispatch's own bounded cudaMalloc block, with owned/borrowed bytes and the reason recorded in
// `native_stats`.  The attention/GDN/shared expert/KV of the layer are the mature Qwen graph's - this file
// only executes the routed experts.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

class ExpertDispatch;

/// Whether this build (and every layer's formats) can run the prompt path on the GPU: MMQ is built and covers
/// the gate/up and down types.  Checked at bridge init; when false the bridge keeps the CPU prompt path.
bool qwen_prefill_mmq_available();

/// One prompt layer's routed experts: T tokens, K routed experts per token, `x_dev` [n_embd, T], `ids_dev`
/// [K, T] (`ids_nb1` bytes between token columns - llama's argsort view is strided), `weights_dev` [1, K, T]
/// (`w_nb1`/`w_nb2` its strides), `out_dev` [n_embd, T] (all device, on `stream`'s device).  Returns false
/// with `err`; the counters live in `ExpertDispatch::native_stats`.
bool qwen_prefill_experts(ExpertDispatch& d, int64_t layer, int64_t T, const void* x_dev, int64_t x_nb1,
                          const void* ids_dev, int64_t ids_nb1, const void* weights_dev, int64_t w_nb1,
                          int64_t w_nb2, void* out_dev, void* stream, std::string& err);

/// The prompt -> decode transition: waits for the prompt's device work, refills the lent cache slots from the
/// source and restores their residency (no-op when the workspace was owned).  The runner calls it after the
/// prompt loop; `qwen_prefill_close` calls it again as a safety net.  False with `err` when a refill failed
/// (those slots stay `kNotResident`: the decode tier sees a miss, never a fake hit).
bool qwen_prefill_loan_return(ExpertDispatch& d, std::string& err);

/// Frees the prompt path's buffers, ring and MMQ context (process teardown).  Device buffers inside a borrowed
/// region are left to the cache (never cudaFree'd).
void qwen_prefill_close(ExpertDispatch& d);

}  // namespace strata::core
