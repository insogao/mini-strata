// src/prefill/cold_stager.cpp - the ColdStager methods (see include/strata/prefill/cold_stager.hpp): the
// file-local `Stager` of src/prefill/prefill.cpp (method bodies as moved out, plus the opt-in error/lifecycle
// mode used by the Qwen prompt path; the legacy path is unchanged).
#include "strata/prefill/cold_stager.hpp"

#include "strata/core/expert_source.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Real hardware store fence for the write-combined payload ring (STRATA_QWEN35_STAGER_WC=1 only): the producer
// must retire every payload store before it publishes `ready`, and a plain C++ release store cannot order the
// weak WC writes.  x86/x64: SFENCE (Intel SDM: drains the write-combining buffers).  Other platforms are
// refused at init - never silently assume an equivalent fence.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define STRATA_STAGER_WC_FENCE() _mm_sfence()
#define STRATA_STAGER_WC_PLATFORM 1
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <xmmintrin.h>
#define STRATA_STAGER_WC_FENCE() _mm_sfence()
#define STRATA_STAGER_WC_PLATFORM 1
#else
#define STRATA_STAGER_WC_FENCE() ((void) 0)
#define STRATA_STAGER_WC_PLATFORM 0
#endif

namespace strata::prefill {
namespace {

// The same semantics as the file-local helpers in src/prefill/prefill.cpp (which keeps them for its other call
// sites): STRATA_TEST_PAGEABLE=1 fails every pinned host allocation in init (the pageable fallbacks' ASan runs),
// and STRATA_STAGER_SLEEP=1/0 forces the threads' sleeping waits either way (#1057: default on Linux, spin on
// Windows).
inline bool force_pageable() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline bool stager_sleep() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_STAGER_SLEEP");
        if (e != nullptr && e[0] != '\0') return e[0] != '0';
#if defined(_WIN32)
        return false;
#else
        return true;
#endif
    }();
    return v;
}

// finish() and cancel() publish this as `issued`: it is larger than any job index, so an issued wait can never
// sleep on after a failure, and a normal finish wakes every waiter.
constexpr int kIssuedDone = 1 << 30;

// Host wall for the opt-in perf scopes; only called inside `if (perf)` branches (default off: no clock).
inline uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

bool ColdStager::init(size_t bytes, int nthreads) {
    // The Flash-Next call sites: env ring, env/platform sleep, pageable fallback, abort on a source failure.
    return init(bytes, nthreads, ColdStagerPolicy{});
}

bool ColdStager::init(size_t bytes, int nthreads, const ColdStagerPolicy& p) {
    if (inited) return false;   // one init per instance: the destructor sizes its cleanup on what init created
    if (nthreads < 1) return false;
    inited = true;
    blob_bytes = bytes;
    opt = p.propagate_error;
    strict_pin = p.strict_pinned;
    perf = p.perf_counters;
    wc = p.write_combined;
    if (wc && !STRATA_STAGER_WC_PLATFORM) {
        std::fprintf(stderr, "strata: stager write-combined requested but this platform/compiler has no verified "
                             "hardware store fence; refusing the WC ring\n");
        return false;
    }
    sleep_mode = p.sleep == -1 ? stager_sleep() : (p.sleep != 0);
    if (p.ring > 0) kRing = std::clamp(p.ring, 2, 256);
    else if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
    buf.assign((size_t) kRing, nullptr);
    pinned.assign((size_t) kRing, 0);
    dma_done.assign((size_t) kRing, nullptr);
    pageable.resize((size_t) kRing);
    unsigned int wc_flags = 0;
    const unsigned int alloc_flags = wc ? (cudaHostAllocDefault | cudaHostAllocWriteCombined) : cudaHostAllocDefault;
    for (int i = 0; i < kRing; ++i) {
        const bool want_pin = p.strict_pinned || !force_pageable();
        pinned[i] = want_pin && cudaHostAlloc((void**) &buf[i], bytes, alloc_flags) == cudaSuccess;
        if (pinned[i] && wc) {
            // WC requested: verify the driver really returned write-combined memory; a cacheable buffer would
            // otherwise silently pass as WC.  A verification failure refuses the ring (strict-pin cleanup frees
            // the buffer through the destructor).
            unsigned int flags = 0;
            if (cudaHostGetFlags(&flags, buf[i]) != cudaSuccess || (flags & cudaHostAllocWriteCombined) == 0) {
                std::fprintf(stderr, "strata: stager write-combined buffer %d is not write-combined (flags 0x%x); "
                                     "refusing the WC ring\n", i, flags);
                return false;
            }
            if (i == 0) wc_flags = flags;
        }
        if (!pinned[i]) {
            if (strict_pin || wc) return false;   // opt-in: pinned or fail; WC never falls back to pageable/cached
            cudaGetLastError();
            pageable[(size_t) i].resize(bytes);
            buf[i] = pageable[(size_t) i].data();
        } else {
            ++pinned_count;
        }
        if (cudaEventCreateWithFlags(&dma_done[i], sleep_mode ? (cudaEventDisableTiming | cudaEventBlockingSync)
                                                              : cudaEventDisableTiming) != cudaSuccess) return false;
    }
    if (cudaGetDevice(&device) != cudaSuccess && opt) return false;
    try {
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
    } catch (...) {
        return false;   // the destructor joins whatever was created and frees by the actual sizes/flags
    }
    if (wc) {
        // The one bounded WC evidence line (startup only): the verified buffer flags, the real pinned bytes and
        // the producer count.  The metadata (ready/head/issued/atomics) is ordinary cacheable memory.
        std::fprintf(stderr, "strata: stager write-combined ring: buffers %d x %zu B, verified flags 0x%x, "
                             "pinned_bytes %zu, threads %d\n",
                     kRing, bytes, wc_flags, host_pinned_bytes(), nthreads);
    }
    return true;
}

ColdStager::~ColdStager() {
    finish();
    { std::lock_guard<std::mutex> lk(mu); quit = true; }
    cv.notify_all();
    for (auto& t : threads) t.join();
    for (size_t i = 0; i < dma_done.size(); ++i) {
        if (dma_done[i] != nullptr) cudaEventDestroy(dma_done[i]);
        if (i < buf.size() && buf[i] != nullptr && i < pinned.size() && pinned[i]) cudaFreeHost(buf[i]);
    }
}

void ColdStager::work() {
    if (cudaSetDevice(device) != cudaSuccess) {
        if (opt) { cancel("the producer could not select its CUDA device"); return; }
    }
    uint32_t seen = 0;
    uint64_t wake_start_us = 0;
    bool first_done = true;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return quit || gen != seen; });
            if (quit) return;
            seen = gen;
        }
        if (perf) {
            // Perf only: one wake per generation; the start timestamp was published with the head release, so
            // it is ordered for this `seen` (no other start can run before this generation finishes).
            perf_wakes.fetch_add(1, std::memory_order_relaxed);
            wake_start_us = perf_start_us.load(std::memory_order_relaxed);
            first_done = false;
        }
        for (;;) {
            active.fetch_add(1, std::memory_order_acq_rel);
            // pp-opt: a run of up to `batch` consecutive jobs (batch <= kRing, so the earliest unfinished run
            // never waits for a buffer of its own)
            int k = 1;
            const int j = (opt && failed_flag.load(std::memory_order_acquire)) ? -1 : claim(seen, batch, k);
            if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
            if (perf && !first_done) {
                // start() -> first successful claim of this generation (wake + claim, host wall)
                first_done = true;
                const uint64_t nw = now_us();
                perf_first_us.fetch_add(nw > wake_start_us ? nw - wake_start_us : 0, std::memory_order_relaxed);
                perf_first_n.fetch_add(1, std::memory_order_relaxed);
            }
            bool bad = false;
            for (int i = j; i < j + k && !bad; ++i) {
                // The copies run ahead of the DMAs, so most of these threads spend most of a prompt waiting here: sleep
                // (atomic wait, and a blocking-sync event below), not a yield spin - 32 spinners took every core
                // (Linux; see stager_sleep for Windows).
                if (i >= kRing) {   // job i - kRing's DMA from this buffer is queued
                    const uint64_t gt0 = perf ? now_us() : 0;
                    if (perf) perf_issued_entries.fetch_add(1, std::memory_order_relaxed);
                    if (sleep_mode)
                        for (int x; (x = issued.load(std::memory_order_acquire)) <= i - kRing;) {
                            if (opt && failed_flag.load(std::memory_order_acquire)) break;
                            issued.wait(x);
                        }
                    else
                        while (issued.load(std::memory_order_acquire) <= i - kRing) {
                            if (opt && failed_flag.load(std::memory_order_acquire)) break;
                            std::this_thread::yield();
                        }
                    if (perf) perf_issued_us.fetch_add(now_us() - gt0, std::memory_order_relaxed);
                    if (opt && failed_flag.load(std::memory_order_acquire)) { bad = true; break; }
                }
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                const uint64_t dt0 = perf ? now_us() : 0;
                const cudaError_t dme = cudaEventSynchronize(dma_done[i % kRing]);
                if (perf) {
                    perf_dma_calls.fetch_add(1, std::memory_order_relaxed);
                    perf_dma_us.fetch_add(now_us() - dt0, std::memory_order_relaxed);
                }
                if (dme != cudaSuccess && opt) {
                    cancel("the producer's dma_done wait failed");
                    bad = true;
                    break;
                }
            }
            if (!bad) {
                const uint64_t ct0 = perf ? now_us() : 0;
                int32_t rl[kMaxBatch], re[kMaxBatch];
                uint8_t* rd[kMaxBatch];
                size_t nr = 0;
                core::ExpertSource* from = nullptr;
                uint64_t cjobs = 0, cbytes = 0;
                for (int i = j; i < j + k; ++i) {
                    if (opt && failed_flag.load(std::memory_order_acquire)) { bad = true; break; }
                    const Job& jb = jobs[(size_t) i];
                    if (jb.from == nullptr) {
                        std::memcpy(buf[i % kRing], jb.src, jb.bytes);
                        if (perf) { ++cjobs; cbytes += jb.bytes; }
                        continue;
                    }
                    from = jb.from;
                    rl[nr] = jb.l; re[nr] = jb.e; rd[nr] = buf[i % kRing]; ++nr;
                    if (perf) { ++cjobs; cbytes += jb.bytes; }
                }
                if (!bad && nr > 0 && !from->copy_blobs(rl, re, rd, nr)) {
                    if (opt) {
                        char m[160];
                        std::snprintf(m, sizeof(m), "the expert source could not copy expert %d of layer %d", re[0], rl[0]);
                        cancel(m);
                        bad = true;
                    } else {
                        std::fprintf(stderr, "prefill: the expert source could not copy experts %d.. of layer %d\n", re[0], rl[0]);
                        std::abort();
                    }
                }
                if (perf) {
                    // The attempted copy scope's host wall (success or failure); jobs/bytes count only a fully
                    // successful batch - a failed/cancelled copy is never recorded as successful bytes.
                    perf_src_us.fetch_add(now_us() - ct0, std::memory_order_relaxed);
                    if (!bad) {
                        perf_src_jobs.fetch_add(cjobs, std::memory_order_relaxed);
                        perf_src_bytes.fetch_add(cbytes, std::memory_order_relaxed);
                    }
                }
            }
            if (wc && !bad) {
                // REAL hardware store fence (WC only): every payload store of this run (the memcpy branch and
                // the source's copy_blobs into the write-combined buffers) must be retired before any `ready`
                // release becomes visible - a plain C++ release store cannot order weak WC writes.
                STRATA_STAGER_WC_FENCE();
            }
            for (int i = j; i < j + k; ++i) ready[(size_t) i].store(bad ? -1 : 1, std::memory_order_release);
            active.fetch_sub(1, std::memory_order_acq_rel);
            if (bad) break;   // stop claiming; the next cv wake (quit) ends the thread
        }
    }
}

int ColdStager::claim(uint32_t g, int want, int& k) {
    uint64_t cur = head.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (cur >> 32) != g) return -1;
        const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
        if (j >= n) return -1;
        k = std::min(want, n - j);
        if (head.compare_exchange_weak(cur, cur + (uint64_t) k, std::memory_order_acq_rel, std::memory_order_acquire))
            return j;
    }
}

void ColdStager::start(std::vector<Job>&& js) {
    if (js.empty()) return;
    std::lock_guard<std::mutex> lk(mu);
    jobs = std::move(js);
    if (ready_cap < jobs.size()) {
        ready_cap = jobs.size() * 2;
        ready.reset(new std::atomic<int>[ready_cap]);
    }
    for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
    issued.store(0);
    ++gen;
    stat_starts.fetch_add(1, std::memory_order_relaxed);
    stat_jobs.fetch_add(jobs.size(), std::memory_order_relaxed);
    // Perf only: timestamp published BEFORE the head release-store below; the worker reads it after its head
    // acquire (claim) for the matching generation, so the value is ordered and never torn (atomic store).
    if (perf) perf_start_us.store(now_us(), std::memory_order_relaxed);
    head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
    cv.notify_all();
}

const uint8_t* ColdStager::wait(int j) {
    for (;;) {
        const int r = ready[(size_t) j].load(std::memory_order_acquire);
        if (r == 1) return buf[j % kRing];
        if (r == -1) return nullptr;   // this batch was abandoned (opt-in failure)
        if (opt && failed_flag.load(std::memory_order_acquire)) return nullptr;
        std::this_thread::yield();
    }
}

bool ColdStager::issued_one(int j, cudaStream_t copy) {
    const cudaError_t e = cudaEventRecord(dma_done[j % kRing], copy);
    if (e != cudaSuccess && opt) {
        // No successful issue: the buffer must NOT look free to the producer (the old capture would authorize
        // overwriting bytes an in-flight DMA is reading).  Latch; the caller stops its host side.
        cancel("a producer dma_done event could not be recorded");
        return false;
    }
    issued.store(j + 1, std::memory_order_release);
    issued.notify_all();
    stat_issued.fetch_add(1, std::memory_order_relaxed);
    return e == cudaSuccess;
}

void ColdStager::finish() {
    head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
    issued.store(kIssuedDone, std::memory_order_release);
    issued.notify_all();
    while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    stat_finishes.fetch_add(1, std::memory_order_relaxed);
}

void ColdStager::cancel(const char* why) {
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(fail_mu);
        if (fail_msg.empty()) {
            fail_msg = why != nullptr ? why : "the expert producer failed";
            first = true;
        }
    }
    failed_flag.store(true, std::memory_order_release);
    // Change the state an issued wait sleeps on: nothing can keep waiting, and no waiter can be released early
    // by a later normal issue (the sentinel is above every job index).
    const int cur = issued.load(std::memory_order_acquire);
    if (cur < kIssuedDone) issued.store(kIssuedDone, std::memory_order_release);
    issued.notify_all();
    if (first) stat_fails.fetch_add(1, std::memory_order_relaxed);
}

const char* ColdStager::fail_text() const {
    if (!failed_flag.load(std::memory_order_acquire)) return "";
    std::lock_guard<std::mutex> lk(fail_mu);
    return fail_msg.c_str();   // written once, before the latch: stable after the first failure
}

}  // namespace strata::prefill
