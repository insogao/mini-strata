// include/strata/kernels/cpu/pool.hpp - P2.S3: the CPU expert pool.
//
// A layer runs TEN experts against ONE activation, and the expert kernel is DRAM-bound (Memory/LEDGER.md L9:
// 42.55 GB/s on 6 cores, tracking core count almost exactly).  So the pool's job is not to be clever - it is
// to keep every physical core reading expert bytes for the whole layer, and to be cheap enough that ten
// dispatches per layer cost less than one expert.
//
// WHY A FLAT BATCH AND NOT A RING.  P2.S3 describes a "lock-free SPMC ring", which is what you need when
// jobs arrive while others are still being computed.  Here they cannot: the host must SUM all ten outputs
// before the next layer starts, so a layer is a barrier by construction and the queue never holds more than
// one batch.  A ring would add a wrap-around to get wrong and buy nothing.  What is kept from the phase is
// the part that matters: `head`/`done` are single fetch_add counters, one claim per worker, no lock.
//
// THE COMPLETION PROTOCOL, because this is where a pool usually goes wrong.  `run()` waits for `done == n`
// AND for every worker to PARK.  Waiting only for `done` is not enough: a worker can still be inside the
// drain loop after its last `done` increment, and the host resetting `head` underneath it would let that
// worker claim a job from the NEXT batch before the next batch has been published.
//
// **AND `parked` IS NOT ENOUGH EITHER (issue #29).**  A worker that went to sleep (after `kSpinBeforeSleep`) is
// still counted as parked when it wakes, so for a moment after it has seen a new epoch the host believes it is
// idle.  On a card with most experts in VRAM the workers sleep in the middle of a request, tiny batches finish
// before a sleeper is awake, and that moment comes round constantly: a late worker could claim from the NEXT
// batch while the host was still writing it, run a job twice, or add to `done` after the host had reset it -
// and `done != n` then never ended, with the GPU waiting on the pool forever.  So every claim carries its
// batch: `head` is one 64-bit word `epoch | njobs | index`, a claim is a CAS that only succeeds for the epoch
// the worker woke for, and a late worker's claim simply fails.  A claim that succeeds belongs to the current
// batch, whose description the host cannot change until that job's `done` has landed.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <atomic>
#include <cstdio>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

/// One expert evaluation.  `act` is SHARED and read-only across the whole batch - that sharing is the point
/// of `s2_expert_vnni_q` and it is what saves 480 redundant activation conversions per token.
struct ExpertJob {
    const uint8_t* blob = nullptr;   ///< one 1,382,400-byte expert
    const ActQ* act = nullptr;       ///< the layer's quantized activation, shared
    float* out = nullptr;            ///< H floats, written by exactly one worker
    float weight = 1.0f;             ///< the router weight; applied by the HOST, not here
    int slot = -1;                   ///< the job's index, for diagnostics
};

/// Plan v0.3 P6: one expert for the `nt` tokens of a verify window that were routed to it.  Every token's output
/// is bitwise the single-token job's.
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const ActQ* act[MAXT] = {};
    float* out[MAXT] = {};
    /// Plan v0.3 P6: a native pack's activations (the layer's `vec_dot_type`), one per token.
    const void* nact[MAXT] = {};
};

/// The Qwen35 graph path (native pack read in place): one expert's three GGUF role slices against up to MAXT
/// quantized activations (all tokens of a prefill tile routed to it), the whole expert in one worker task -
/// gate/up rows, SwiGLU, hidden quantization, down rows.  The row-major loops read each weight row once for
/// every activation (the slices avoid the staging copy `native_gu_rows` would otherwise read); a packed `blob`
/// (an experts.bin pack) is the fallback.  `ff` and `hq` are the job's own scratch, one per activation.
struct NativeSliceJob {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    const uint8_t* blob = nullptr;   ///< packed fallback when the source has no role slices
    int nt = 0;
    const void* act[MAXT] = {};      ///< this job's quantized activations (the layer's gu_act)
    float* ff[MAXT] = {};            ///< n_ff floats each
    void* hq[MAXT] = {};             ///< the layer's d_act over n_ff
    float* out[MAXT] = {};           ///< n_embd floats each
};

/// How worker threads are allocated across physical/logical CPU cores (#272).  `All` is the layout the pool has
/// always used and the default; the hybrid-aware ones are opt-in (--pool-affinity auto|p-cores).
enum class PoolAffinity {
    Auto,      ///< Hybrid: prioritize physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Restrict workers strictly to Performance cores and their SMT siblings
    All,       ///< The default: one worker per physical core in the OS's order, without hybrid distinction
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Total logical threads on performance cores
    int e_cores = 0;                ///< Efficient cores
    std::vector<int> worker_cores;  ///< Ordered CPU IDs; on Windows, group * 64 + processor within the group
    int host_core = -1;             ///< Logical core reserved for host thread
};

/// Which core the host thread - the layer loop that spins on the GPU's flags - takes when `skip_first` reserves one
/// (--host-core first|last, STRATA_HOST_CORE).  `First` is the layout the pool has always used.  `Last` (eddoursul's
/// fork, F12) puts the host on the last physical core and lets the workers have the first: Windows sends a GPU's
/// interrupts to one logical processor, usually the first, and every copy that lands raises one (~13 us in the ISR
/// and a DPC), which delays a spinning host there.  The placement changes no result, only where threads run.  A hybrid
/// CPU keeps `First` (its first core is the best P-core).
enum class HostCore { First, Last };
void set_host_core(HostCore where);
HostCore host_core_setting();

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// The logical processor the host thread is pinned to under the current setting (-1: no topology).
int planned_host_core(PoolAffinity affinity = PoolAffinity::All);

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another
/// worker.  On the 6-core/12-thread machine this project measures on, `hardware_concurrency()/2` workers on
/// logical processors 0..5 would put every worker on a sibling pair and halve the useful bandwidth - which is
/// exactly the kind of error that shows up as "the CPU path is slower than the model says" with no clue why.
///
/// `skip_first` drops the first core, which P2.S3 reserves for the host loop.
std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// **THE RESERVATION IS A FICTION UNLESS THE HOST IS ACTUALLY PUT THERE.**
///
/// `physical_cores(true)` keeps the workers off the first physical core so that the host loop can spin on
/// `cudaEventQuery` without stealing a worker's cycles.  Nothing in the pool can enforce the other half of
/// that, so this is it: the host loop calls this on entry and restores on exit.
///
/// MEASURED, and this is why it exists: the pool runs at **36.32 GB/s on 5 workers with nothing else running**
/// - exactly 5/6 of L9's 44.14 on 6 - and at **26.9 GB/s inside the host loop**, where the unpinned spinning
/// host is free to land on a worker's core or its SMT sibling.  That 1.35x is not the kernel.
///
/// The previous host placement. `valid` is false when querying or setting placement failed.
/// Windows uses a reversible CPU Set selection, leaving hard affinity (including Windows 11's implicit
/// all-group eligibility) untouched. An empty selection restores inheritance from the process defaults.
/// Linux stores the complete, dynamically sized native CPU mask, including CPU IDs above 63.
struct ThreadAffinity {
#if defined(_WIN32)
    std::vector<unsigned long> cpu_sets;
#else
    std::vector<unsigned long> mask;
#endif
    bool valid = false;
};

/// Selects one encoded processor for the caller; Windows CPU Sets respect existing hard-affinity limits.
/// Restore on the same calling thread. Pool-owned workers use hard group affinity instead.
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);

/// Task 202/204: bounded, same-clock HOST subphase timing of one `run_native_slice_jobs` call (request 2,
/// actual decode evaluations 6..8 only, driven by the caller).  Namespace scope (task 204 repair: it used to
/// be nested in ExpertPool while the headers referenced it as strata::kernels::cpu::SlicePhaseTiming).  All
/// values are millis within the current call; `available=false` with a reason code never fabricates a phase.
/// The inclusive spans match the existing `ms_multi_*` semantics for this call; the five phase-8/9 boundaries
/// are the caller's wait_parked(before) / publish / host_drain / wait_done / wait_parked(after) segments.
/// `drain` includes host work, `wait_done` includes the workers' remaining compute, and the park checks are
/// not necessarily wake-ups.
struct SlicePhaseTiming {
    static constexpr int kReasonOk = 0;
    static constexpr int kReasonSingleThreadInline = 1;
    bool requested = false;      ///< the caller asked for timing (gates + an open HOST record)
    bool available = false;      ///< the pool filled the fields (false + reason otherwise)
    int reason = kReasonSingleThreadInline;
    int64_t n_jobs = 0;
    double gu_inclusive_ms = -1, quant_ms = -1, down_inclusive_ms = -1;
    double gu_before_park_ms = -1, gu_publish_ms = -1, gu_host_drain_ms = -1,
           gu_wait_done_ms = -1, gu_after_park_ms = -1;
    double down_before_park_ms = -1, down_publish_ms = -1, down_host_drain_ms = -1,
           down_wait_done_ms = -1, down_after_park_ms = -1;
};

class ExpertPool {
public:
    /// `n_workers <= 0` means "every physical core except the first".  Workers are pinned to physical cores
    /// (minus core 0 by default) and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// **`host_works` PUTS THE HOST THREAD INTO THE DRAIN (R2.2's FIRST HALF).**
    ///
    /// The pool reserves core 0 for the host loop so the doorbell spin cannot steal a worker's cycles - but
    /// during `run()` the host does not spin, it waits, so core 0 is idle for the whole drain. Measured on the
    /// 6-core machine this project targets: the engine's pool drains at **33.7 GB/s** (663.6 MB of expert
    /// blobs in 19.71 ms/token) where the same kernel on 5 workers should reach 5/6 x 44.14 = 36.8 and the
    /// machine measures 44.14 GB/s on all six. So the sixth core is being paid for and not used.
    ///
    /// With `host_works`, `run()` claims jobs itself instead of spinning on `done_`, and the pool is six
    /// threads on six cores. `false` is the A/B arm and exists so the change is measurable rather than
    /// asserted - the counter it moves is `pool phases ... drain`, which is host-side and needs no profiler.
    explicit ExpertPool(int n_workers = 0, bool pin = true, bool host_works = true,
                        PoolAffinity affinity = PoolAffinity::All, int tasks = 0);
    /// Batched row phases: 0 keeps three tasks per participating thread; explicit counts are capped by rows.
    static constexpr int kMaxTasks = 4096;
    /// The watchdog's view of the pool (issue #31): the batch, the counters, every thread's state.
    void diag(std::FILE* f) const;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    /// Whether the host thread also drains.  Reported at startup, because "the engine adapts to the machine it
    /// is on" is only true if the engine says which adaptation it took.
    bool host_works() const { return host_works_; }

    bool is_hybrid() const { return topo_.is_hybrid; }
    int p_cores() const { return topo_.p_cores; }
    int p_threads() const { return topo_.p_threads; }
    int e_cores() const { return topo_.e_cores; }
    PoolAffinity affinity() const { return affinity_; }

    /// Publish `n` jobs, then block until every one has been claimed AND every worker has parked.
    /// `jobs` must outlive the call (it does, and the workers never touch it afterwards).
    void run(ExpertJob* jobs, int n);

    /// Plan v0.3 P4: the same outputs as `run`, bitwise, with every expert split by rows across all threads
    /// (gate/up rows, then the intermediate's quantization, then down rows).  With fewer experts than threads -
    /// the case once the VRAM tier takes half of them - `run` leaves cores idle and each expert streams at one
    /// core's bandwidth; this streams every expert at all of them.  At most `kMaxSplit` experts.
    void run_split(ExpertJob* jobs, int n);
    static constexpr int kMaxSplit = 16;
    /// Plan v0.3 P6: `run_split` for multi-token jobs (at most `kMaxSplitMulti`); the rows of each expert are
    /// read once for all of its tokens.
    void run_split_multi(ExpertJobMulti* jobs, int n);
    /// Plan v0.3 P6: the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n);
    /// The Qwen35 path: the layer's three GGUF role slices (or a packed blob fallback), ROW-SPLIT across all
    /// threads exactly like `run_split_multi_native`'s phases: gate/up rows, the host's intermediate
    /// quantization, down rows.  Each row still runs the whole-expert task's per-row kernel, so the output is
    /// bitwise the same; with only a few missed experts a whole-expert task would leave most cores idle.
    /// `timing` (task 202, default null) is an OPTIONAL explicit output: when non-null and `requested`, the
    /// phases 8/9 caller-side host boundaries and the same-call inclusive/quant spans are filled; every other
    /// caller keeps the exact previous behaviour (no clock, no allocation, no output).
    void run_native_slice_jobs(const NativeFmt& f, NativeSliceJob* jobs, int n, SlicePhaseTiming* timing = nullptr);
    static constexpr int kMaxSplitMulti = 96;
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;

    /// Total `_mm_pause` iterations spent waiting, over all workers, is no longer counted - see the note on the
    /// atomics below.  It was a LOCKED read-modify-write in the park loop, so measuring the contention added to
    /// it.
    long long pauses() const { return 0; }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime.**  Three phases per
    /// layer - wait for every worker to be parked, wait for the drain, wait for them to re-park - and until now
    /// all three were reported as one number.  Without the split there is no way to tell a pool that is slow at
    /// the WORK from one that is slow at the SYNCHRONISATION, and those need opposite fixes: the first is a
    /// kernel problem and the second is a barrier problem.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& wait_park, double& drain, double& repark) const {
        wait_park = ms_wait_park_;
        drain = ms_drain_;
        repark = ms_repark_;
    }

    /// **A PARKED WORKER SPINS FOR THIS LONG, THEN SLEEPS.**  The park is a `_mm_pause` spin because a layer's
    /// batches are microseconds apart and a wake-up from the OS costs more than that.  But a spin that never ends
    /// keeps every worker's core at 100% while the engine waits for a request - issue #4, "CPU 50% even when
    /// doing nothing" (7 of the 5700X's 16 threads).  Between requests the workers block on `sleep_cv_`.  NOT only
    /// between requests: with most experts in VRAM (a 24 GB card) many layers have no CPU work, so the workers
    /// also sleep mid-request - which the claim protocol above must survive (issue #29).
    /// `STRATA_POOL_SPIN_US` overrides it (a test knob: a short spin makes the workers sleep constantly).
    static constexpr std::chrono::milliseconds kSpinBeforeSleep{20};
    /// A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of
    /// spinning forever, and the server starts it again (issue #29).
    static constexpr std::chrono::seconds kStall{60};

private:
    int phase_tasks(int64_t rows) const;
    int tasks_ = 0;
    void worker(int id);
    void drain(int id, ExpertScratch& scratch, uint32_t epoch);
    void run_phase(int mode, int n_tasks, SlicePhaseTiming* phase_timing = nullptr);
    /// Claim the next job of batch `epoch`, or -1 (that batch is exhausted, or it is not the current one).
    int claim(uint32_t epoch);
    /// Publish the batch whose description the caller has just written: reset `done`, then `head`, then the epoch.
    uint32_t begin_batch(int n);
    /// The host's waits, bounded by `kStall`.
    void wait_parked(const char* what);
    void wait_done(int n);
    /// Bump `epoch_`, and wake the workers that went to sleep.  Every publish goes through here.
    void publish();

    int n_ = 0;
    bool host_works_ = true;
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch when `host_works_`.  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s three phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_wait_park_ = 0.0;
    double ms_drain_ = 0.0;
    double ms_repark_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE, AND THE SPIN COUNTER IS GONE.  (Review finding C3.)
    //
    // These were six adjacent atomics, which put `head_`, `done_`, `parked_` and `epoch_` on ONE cache line -
    // the four that workers and the host actually contend on, invalidating each other on every access.
    //
    // Worse, the parked spin did `pauses_.fetch_add(1)` on EVERY iteration: a LOCKED read-modify-write, five
    // workers against one line, at roughly one iteration per `_mm_pause`.  So the line the host must WRITE to
    // publish work (`epoch_`) and READ to confirm the workers are parked (`parked_`) was being hammered by the
    // very threads waiting for it.  That is contention the pool imposes on itself.
    //
    // The counter was diagnostic only - `pauses()` was read in one place, to print a number nothing branched on
    // - so it is deleted rather than amortised.  `alignas(64)` then stops the remaining four sharing.
    // issue #31 diagnostics: each worker's state (kParked, kSleeping, kBetween, or the job it runs) and the host's
    // (kIdle, kWaitParked, kWaitDone, or its job), printed by the serve watchdog through `diag`
    static constexpr int32_t kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11,
                             kWaitDone = -12;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::atomic<int64_t> hstate_ms_{0};
    alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index (issue #29)
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<bool> stop_{false};
    // The sleep after `kSpinBeforeSleep`.  `sleepers_` is how `publish` knows whether anyone needs waking, so the
    // token path pays one uncontended load per publish and never takes the mutex while the workers spin.
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::chrono::microseconds spin_before_sleep_{kSpinBeforeSleep};
    std::vector<std::thread> threads_;
    std::vector<ExpertScratch> scratch_;   // one per worker: no allocation, no false sharing of the hot data
    // run_split state: mode 0 = whole experts, 1 = gate/up row parts, 2 = down row parts
    int mode_ = 0;
    int parts_a_ = 1, parts_b_ = 1;
    struct SplitBuf {
        alignas(64) float ff[FF];
        ActQ a2;
    };
    std::vector<SplitBuf> split_;
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;     // rows of the current multi phase across all its experts (n * FF, then n * H)
    int mtasks_ = 1;        // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // plan v0.3 P6: native down activations
    };
    const NativeFmt* nfmt_ = nullptr;
    NativeSliceJob* sjobs_ = nullptr;   ///< mode 7: one whole-expert task per index
    std::vector<SplitBufMulti> split_multi_;
    PoolAffinity affinity_ = PoolAffinity::All;
    CpuTopology topo_;
};

}  // namespace strata::kernels::cpu
