// include/strata/prefill/cold_stager.hpp - the cold experts' host-side stager, moved mechanically out of
// src/prefill/prefill.cpp's file-local `Stager` (its method bodies are in src/prefill/cold_stager.cpp;
// prefill.cpp keeps the name through `using Stager = ColdStager;`) plus an OPT-IN error/lifecycle mode.
//
// Legacy behaviour (init(blob_bytes, nthreads), the Flash-Next call sites): STRATA_STAGER_RING /
// STRATA_STAGER_SLEEP / pageable fallback / std::abort on a source failure - byte-for-byte the original.
// Opt-in mode (the third init overload, used by the Qwen prompt path): an explicit ring/thread budget, strict
// pinning, a failure that is published instead of aborting, and wait() that returns nullptr instead of a
// half-written buffer.  One init per instance; safe to destroy after a failed or partial init.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core { class ExpertSource; }

namespace strata::prefill {

// The opt-in policy (frozen for the Qwen prompt path: ring 16, one producer, strict pinned, sleeping issued
// wait with BlockingSync events, error propagation).  `sleep` must be chosen before init - it selects the
// dma_done event flags.  `perf_counters` is a Qwen-only opt-in: it enables the producer-side host-wall
// counters below (no clocks, no atomics and no output while false; the legacy init never enables them).
struct ColdStagerPolicy {
    int ring = 0;                 // 0: legacy (STRATA_STAGER_RING env, default 16); >0: explicit, clamped 2..256
    int sleep = -1;               // -1: legacy (STRATA_STAGER_SLEEP / platform); 0: spin; 1: sleep
    bool strict_pinned = false;   // true: any host pin failure fails init (no pageable fallback)
    bool propagate_error = false; // true: latch the failure, stop claiming, wait() -> nullptr (no abort)
    bool perf_counters = false;   // true: accumulate the host-wall scopes (source copy / issued gate / dma wait)
    /// Opt-in write-combined payload ring (default false; Qwen only via exact STRATA_QWEN35_STAGER_WC=1):
    /// the kRing payload buffers are cudaHostAllocWriteCombined (host-write/device-read DMA), the metadata
    /// (ready/head/issued/atomics) stays ordinary cacheable memory, and the producer publishes ready only
    /// after a real hardware store fence.  CPU reads of the payload (e.g. the STAGER_CHECK memcmp) conflict
    /// with WC and must be refused by the caller; unsupported platforms are refused at init.
    bool write_combined = false;
};

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct ColdStager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // pp-opt: the jobs a thread claims at once (STRATA_STAGER_BATCH; set by init's caller, at most kMaxBatch and kRing)
    static constexpr int kMaxBatch = 32;
    int batch = 1;
    // `from` set: the blob is copied by the source itself (CS-T: a GGUF read in place assembles it from its three
    // role slices; a pointer to it would not live as long as the queue)
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;
    // opt-in lifecycle/error state (defaults keep the legacy instance unchanged)
    bool inited = false;          ///< init ran once; a retry is refused (the destructor relies on it)
    bool opt = false;             ///< the opt-in mode is in force (error propagation, strict pin, sleeping waits)
    bool strict_pin = false;
    bool sleep_mode = false;      ///< decided at init: dma_done event flags and the issued waits
    bool wc = false;              ///< payload ring is write-combined (decided at init; verified by cudaHostGetFlags)
    size_t blob_bytes = 0;        ///< one buffer's size (what init allocated)
    int pinned_count = 0;         ///< buffers that really are cudaHostAlloc'd (host pinned bytes = this * blob_bytes)
    std::atomic<bool> failed_flag{false};
    mutable std::mutex fail_mu;   ///< guards `fail_msg` (written once, read any time)
    std::string fail_msg;
    // sparse caller-thread stats (start/issued_one/finish/cancel); the opt-in close reports these, no per-job
    // trace.  Atomic: cancel() may run on a producer thread.
    std::atomic<uint64_t> stat_starts{0}, stat_jobs{0}, stat_issued{0}, stat_finishes{0}, stat_fails{0};
    // opt-in producer host-wall scopes (policy.perf_counters, default off; relaxed atomics, written only when
    // enabled).  They are host API wall readings on the producer thread - NOT GPU busy time, not a critical
    // path.  Scopes are sequential per job (issued gate -> dma_done wait -> source copy); claim/ready/issued_one
    // and the between-layer cv wait are not covered; with more workers a sum would overlap.
    bool perf = false;              ///< decided at init from the policy; false adds no clock/atomic work
    std::atomic<uint64_t> perf_src_us{0}, perf_src_jobs{0}, perf_src_bytes{0};   ///< real copy scope, successes
    std::atomic<uint64_t> perf_issued_us{0}, perf_issued_entries{0};             ///< i >= kRing gate scope
    std::atomic<uint64_t> perf_dma_us{0}, perf_dma_calls{0};                     ///< cudaEventSynchronize scope
    std::atomic<uint64_t> perf_wakes{0}, perf_first_us{0}, perf_first_n{0};      ///< per-generation wake + first claim
    std::atomic<uint64_t> perf_start_us{0};  ///< start() timestamp for the current generation (perf only)

    bool init(size_t blob_bytes, int nthreads);
    /// Opt-in: explicit ring (0 keeps the env), sleep decision, strict pin, error propagation.  One init per
    /// instance, nthreads >= 1; false leaves the object safe to destroy (partial init included).
    bool init(size_t blob_bytes, int nthreads, const ColdStagerPolicy& policy);
    ~ColdStager();
    void work();
    int claim(uint32_t g, int want, int& k);
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js);
    /// Job j's bytes, in a pinned buffer (waits for the copy).  nullptr when the producer failed (opt-in).
    const uint8_t* wait(int j);
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.  false means the
    /// record failed and issued did NOT advance (opt-in latches the failure; legacy ignores the result).
    bool issued_one(int j, cudaStream_t copy);
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish();
    /// Opt-in failure entry (any thread): publish the text, latch failed, wake every waiter (the issued state
    /// changes so an issued.wait cannot sleep on), stop further claims.  No-op text after the first failure.
    void cancel(const char* why);
    bool failed() const { return failed_flag.load(std::memory_order_acquire); }
    const char* fail_text() const;
    int active_now() const { return active.load(std::memory_order_acquire); }
    size_t host_pinned_bytes() const { return (size_t) pinned_count * blob_bytes; }
};

}  // namespace strata::prefill
