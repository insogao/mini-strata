#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Task 68 GDN eager event diagnostic (default off; exact STRATA_QWEN35_GDN_EVENTS=1).  The Strata runner
// resets/arms at the real prompt start and flushes exactly once after the short decode output; all three are
// no-ops while the env is unset.  Collected inside ggml_cuda_op_gated_delta_net_impl on the op's real stream.
extern "C" void strata_gdn_events_reset(void);
extern "C" void strata_gdn_events_arm(void);
extern "C" void strata_gdn_events_flush(void);
