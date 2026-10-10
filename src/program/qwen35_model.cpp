// src/program/qwen35_model.cpp - `strata generate`'s Qwen3.6-35B-A3B text POC runner.
//
// Uses the pinned, patched llama.cpp graph for attention/GDN/shared-expert/head and the Strata shared expert
// service for the routed experts (see qwen35_bridge.cpp).  The prompt arrives as token ids (the fixed test
// prompts), so no tokenizer is on the measured path; token pieces for the evidence come from the model vocab.
#include "strata/program/qwen35_model.hpp"

#include "strata/core/qwen35_bridge.hpp"
#include "strata/core/qwen35_prefill.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "llama.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::string json_quote(const std::string& value) {
    std::string result = "\"";
    for (const char ch : value) {
        const auto byte = static_cast<unsigned char>(ch);
        if (byte == '"' || byte == '\\') { result += '\\'; result += ch; }
        else if (byte < 32) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", byte); result += buf; }
        else result += ch;
    }
    return result + '"';
}

int64_t meta_int(const llama_model* model, const char* key, int64_t fallback) {
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) <= 0 || buf[0] == '\0') return fallback;
    return std::strtoll(buf, nullptr, 10);
}

struct VramPeak {
    size_t peak_used = 0;
    size_t total = 0;
    int64_t samples = 0;
    double sample_seconds = 0;   // wall time of every sample() call, successful or not (additive diagnostic)
    double sample() {
        const auto begin = Clock::now();
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            total = total_b;
            const size_t used = total_b - free_b;
            if (used > peak_used) peak_used = used;
            ++samples;
        }
        const double seconds = elapsed(begin);
        sample_seconds += seconds;
        return seconds;
    }
    double mib() const { return (double) peak_used / 1048576.0; }
};

/// NaN is a real failure (a masked -inf is legal and stays): the previous all-NaN logits once exited 0, so
/// the acceptance path rejects NaN itself instead of trusting exit codes or counters.
bool logits_have_nan(const float* logits, int64_t n) {
    for (int64_t i = 0; i < n; ++i)
        if (std::isnan(logits[i])) return true;
    return false;
}

void fill_batch(llama_batch_ext* batch, const llama_token* tokens, int count, int position) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = position + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, count - 1, true);
}

// Graph observation (default off): STRATA_QWEN35_GRAPH_STATS=1 (exact value; anything else keeps the
// baseline logger) counts three fixed CUDA graph DEBUG messages that ggml-cuda.cu already emits and takes
// bounded per-boundary records of the public llama_perf_context n_reused.  It is host-only: no GPU/host
// sync, no compute rerun, no payloads, and every other DEBUG line is dropped exactly as the baseline drops
// it.  The logger callback is a static function over static counters (nothing stack-local is captured), so
// the global logger never dangles; the install survives every early return and the context/model/backend
// teardown (the baseline never restores its logger either).  The counters are atomic because llama/ggml may
// log from worker threads; the records themselves are written from the single request thread only.
bool graph_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_GRAPH_STATS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

std::atomic<uint64_t> g_graph_uid_reused{0};
std::atomic<uint64_t> g_graph_warmup_complete{0};
std::atomic<uint64_t> g_graph_warmup_reset{0};
bool g_graph_n_reused_first_set = false;   // main thread only (one request, one reader)
int32_t g_graph_n_reused_first = 0;

void graph_stats_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_DEBUG) {
        if (text != nullptr) {
            // Exact fixed messages from ggml-cuda.cu: "CUDA Graph id %zu reused" (uid fast path),
            // "%s: CUDA graph warmup complete" (capture intent/transition) and "%s: CUDA graph warmup
            // reset" (property change -> direct execution).  None of them proves a replay happened.
            if (std::strstr(text, "CUDA graph warmup complete") != nullptr) {
                g_graph_warmup_complete.fetch_add(1, std::memory_order_relaxed);
            } else if (std::strstr(text, "CUDA graph warmup reset") != nullptr) {
                g_graph_warmup_reset.fetch_add(1, std::memory_order_relaxed);
            } else if (std::strstr(text, "CUDA Graph id ") != nullptr && std::strstr(text, "reused") != nullptr) {
                g_graph_uid_reused.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return;
    }
    std::fputs(text, stderr);
}

// One compact stderr record per existing boundary (independent metadata; the stdout/summary schema stays
// unchanged): the request boundary ("prompt_end", output_index -1, never reads perf) and each generated
// output ("output", with the public n_reused and the cumulative delta from the first perf-bearing record).
// At most max_new + 1 records.  ctx == nullptr means "do not read perf" (early failure or no eval yet).
void graph_stats_record(const char* at, long long output_index, const llama_context* ctx) {
    if (!graph_stats_enabled()) return;
    const unsigned long long uid_reused = g_graph_uid_reused.load(std::memory_order_relaxed);
    const unsigned long long warmup_complete = g_graph_warmup_complete.load(std::memory_order_relaxed);
    const unsigned long long warmup_reset = g_graph_warmup_reset.load(std::memory_order_relaxed);
    long long n_reused = -1;
    long long n_reused_delta = -1;
    if (ctx != nullptr) {
        const llama_perf_context_data perf = llama_perf_context(ctx);
        n_reused = (long long) perf.n_reused;
        if (!g_graph_n_reused_first_set) {
            g_graph_n_reused_first_set = true;
            g_graph_n_reused_first = perf.n_reused;
        }
        n_reused_delta = (long long) perf.n_reused - (long long) g_graph_n_reused_first;
    }
    std::fprintf(stderr,
                 "strata graph stats: at=%s output_index=%lld uid_reused=%llu warmup_complete=%llu "
                 "warmup_reset=%llu n_reused=%lld n_reused_delta=%lld\n",
                 at, output_index, uid_reused, warmup_complete, warmup_reset, n_reused, n_reused_delta);
}

}  // namespace

int qwen35_run(const Qwen35RunOptions& o) {
    if (o.model.empty() || o.pack_dir.empty() || (o.tokens.empty() && o.prompt_text.empty())) {
        std::fprintf(stderr, "strata generate: the Qwen35 path needs --qwen35-model, --qwen35-pack and a token or text prompt\n");
        return 2;
    }
    // Task 52 diagnostic replay (default off): a capture directory replays ONE layer through the same expert
    // call without any llama context or dense model load.
    if (const char* replay = std::getenv("STRATA_QWEN35_LAYER_REPLAY"); replay != nullptr && replay[0] != '\0') {
        const char* mix = std::getenv("STRATA_QWEN35_LAYER_REPLAY_MIXED");
        return qwen35_layer_replay(o, replay, mix != nullptr && std::strcmp(mix, "1") == 0);
    }
    const auto process_start = Clock::now();
    llama_backend_init();
    // The engine's debug log (llama_graph_n_input_tensors and friends) is thousands of lines per graph; keep
    // the measured runs' I/O comparable to the stock baseline's.
    if (graph_stats_enabled()) {
        // Default-off graph observation wrapper: identical DEBUG dropping, plus the fixed message counters.
        llama_log_set(graph_stats_log, nullptr);
    } else {
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level == GGML_LOG_LEVEL_DEBUG) return;
            std::fputs(text, stderr);
        }, nullptr);
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr && !o.cuda_backend.empty()) {
        if (ggml_backend_load(o.cuda_backend.c_str()) == nullptr) {
            std::fprintf(stderr, "strata generate: cannot load the CUDA backend %s\n", o.cuda_backend.c_str());
            return 1;
        }
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        std::fprintf(stderr, "strata generate: no GPU backend is registered (this build has no GGML_CUDA; pass "
                             "--qwen35-cuda-backend <ggml-cuda.dll>)\n");
        return 1;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(nullptr, llama_model_free);
    {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = (int32_t) o.ngl;
        // `--cpu-moe` keeps the routed expert tensors on the CPU: the equivalent override is the CPU DEVICE's
        // buffer type (the plain ggml_backend_cpu_buffer_type() is a different, non-device buffer and changes
        // how llama places/see the host weights).
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev)
                                                                 : ggml_backend_cpu_buffer_type();
        const llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps.*", cpu_buft},
            {nullptr, nullptr},
        };
        mp.tensor_buft_overrides = overrides;
        model.reset(llama_model_load_from_file(o.model.c_str(), mp));
    }
    if (model == nullptr) {
        std::fprintf(stderr, "strata generate: cannot load %s\n", o.model.c_str());
        return 1;
    }
    {
        char arch[128] = {0};
        if (llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch)) <= 0 ||
            std::strcmp(arch, "qwen35moe") != 0) {
            std::fprintf(stderr, "strata generate: %s is '%s', not a qwen35moe model\n", o.model.c_str(), arch);
            return 2;
        }
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    const int64_t n_layers = llama_model_n_layer(model.get());
    const int64_t n_embd = llama_model_n_embd(model.get());
    const int64_t n_ff = meta_int(model.get(), "qwen35moe.expert_feed_forward_length", 0);
    const int64_t n_expert = meta_int(model.get(), "qwen35moe.expert_count", 0);
    const int64_t n_used = meta_int(model.get(), "qwen35moe.expert_used_count", 0);
    if (n_layers <= 0 || n_embd <= 0 || n_ff <= 0 || n_expert <= 0 || n_used <= 0) {
        std::fprintf(stderr, "strata generate: the model does not carry the qwen35moe MoE metadata (layers %lld, "
                             "embd %lld, ff %lld, experts %lld, used %lld)\n", (long long) n_layers,
                     (long long) n_embd, (long long) n_ff, (long long) n_expert, (long long) n_used);
        return 2;
    }
    // The prompt: fixed token ids, or text tokenized with the model's own vocab (the independent calibration).
    std::vector<llama_token> prompt;
    if (!o.tokens.empty()) {
        prompt.resize(o.tokens.size());
        for (size_t i = 0; i < o.tokens.size(); ++i) {
            if (o.tokens[i] < 0 || o.tokens[i] >= llama_vocab_n_tokens(vocab)) {
                std::fprintf(stderr, "strata generate: prompt token %lld is out of the model's vocab\n",
                             (long long) o.tokens[i]);
                return 2;
            }
            prompt[i] = (llama_token) o.tokens[i];
        }
    } else {
        const int count = -llama_tokenize(vocab, o.prompt_text.data(), o.prompt_text.size(), nullptr, 0, true, true);
        if (count <= 0) {
            std::fprintf(stderr, "strata generate: the prompt text did not tokenize\n");
            return 2;
        }
        prompt.resize((size_t) count);
        if (llama_tokenize(vocab, o.prompt_text.data(), o.prompt_text.size(), prompt.data(), prompt.size(), true,
                           true) != count) {
            std::fprintf(stderr, "strata generate: the prompt text tokenization is inconsistent\n");
            return 2;
        }
    }
    if ((int64_t) prompt.size() + o.max_new > o.max_context) {
        std::fprintf(stderr, "strata generate: the prompt and --max-new do not fit --max-context\n");
        return 2;
    }
    // Optional fixed teacher prefix (perf attribution: the same route trajectory every arm).
    std::vector<llama_token> teacher;
    if (!o.teacher_file.empty()) {
        std::ifstream input(o.teacher_file);
        if (!input) {
            std::fprintf(stderr, "strata generate: cannot open the teacher file %s\n", o.teacher_file.c_str());
            return 2;
        }
        int64_t v = 0;
        while (input >> v) {
            if (v < 0 || v >= llama_vocab_n_tokens(vocab)) {
                std::fprintf(stderr, "strata generate: teacher token %lld is out of vocab\n", (long long) v);
                return 2;
            }
            teacher.push_back((llama_token) v);
        }
        if ((int64_t) teacher.size() < o.max_new) {
            std::fprintf(stderr, "strata generate: the teacher file holds %zu tokens, --max-new needs %lld\n",
                         teacher.size(), (long long) o.max_new);
            return 2;
        }
    }
    VramPeak vram;
    vram.sample();
    std::fprintf(stdout, "{\"event\":\"config\",\"model\":%s,\"pack\":%s,\"layers\":%lld,\"n_embd\":%lld,"
                         "\"n_ff\":%lld,\"experts\":%lld,\"top\":%lld,\"cache_slots\":%lld,\"ngl\":%lld,"
                         "\"threads\":%lld,\"context\":%lld,\"prompt_tokens\":%zu,\"max_new\":%lld,"
                         "\"qwen35_ubatch\":%lld,\"cuda_backend\":%s}\n",
                 json_quote(o.model).c_str(), json_quote(o.pack_dir).c_str(), (long long) n_layers,
                 (long long) n_embd, (long long) n_ff, (long long) n_expert, (long long) n_used,
                 (long long) o.cache_slots, (long long) o.ngl, (long long) o.threads, (long long) o.max_context,
                 prompt.size(), (long long) o.max_new, (long long) o.ubatch,
                 json_quote(o.cuda_backend).c_str());
    std::fflush(stdout);
    core::Qwen35BridgeConfig cfg;
    cfg.model = o.model;
    cfg.pack_dir = o.pack_dir;
    cfg.n_layers = n_layers;
    cfg.n_expert = n_expert;
    cfg.n_embd = n_embd;
    cfg.n_ff = n_ff;
    cfg.k = n_used;
    cfg.cache_slots = o.cache_slots;
    cfg.profile = o.profile;
    cfg.routing_log = o.routing_log;
    cfg.pool_workers = o.pool_workers;
    std::string err;
    if (!core::qwen35_bridge_init(cfg, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) o.max_context;
    cp.n_batch = (uint32_t) (std::max<int64_t>(512, (int64_t) prompt.size()));
    cp.n_ubatch = (uint32_t) o.ubatch;
    cp.n_threads = (int32_t) o.threads;
    cp.n_threads_batch = (int32_t) o.threads;
    cp.op_offload = true;    // llama/llama-server default; the CPU custom prefill op is still pinned by its host weights
    cp.n_seq_max = 1;
    // fill_batch flags exactly one output (its last token), so every process call carries one output row;
    // capping the context's worst-case output reserve at 1 keeps the reserve graph's vocab-projection
    // columns at the real contract instead of n_batch.  The decode path itself is unchanged.
    cp.n_outputs_max = 1;
    cp.no_perf = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (ctx == nullptr) {
        std::fprintf(stderr, "strata generate: context init failed\n");
        core::qwen35_bridge_shutdown();
        return 1;
    }
    // Requested vs actual micro-batch: the request is the CLI value, the actual is the context's own answer
    // through the public llama_n_ubatch (never a re-print of the request).
    std::fprintf(stdout, "{\"event\":\"context_config\",\"requested_ubatch\":%lld,\"n_ubatch\":%u}\n",
                 (long long) o.ubatch, llama_n_ubatch(ctx.get()));
    std::fflush(stdout);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),
                                                                             llama_batch_ext_free);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(),
                                                                          llama_sampler_free);
    if (batch == nullptr || sampler == nullptr) {
        std::fprintf(stderr, "strata generate: batch/sampler init failed\n");
        core::qwen35_bridge_shutdown();
        return 1;
    }
    // Past graph reservation: decode-graph nodes are the external producer from here on.
    if (std::getenv("STRATA_QWEN35_NO_SEAM") == nullptr) core::qwen35_arm_decode_seam(true);
    std::fprintf(stdout, "{\"event\":\"prompt_tokens\",\"ids\":[");
    for (size_t i = 0; i < prompt.size(); ++i) std::fprintf(stdout, "%s%d", i ? "," : "", prompt[i]);
    std::fprintf(stdout, "]}\n");
    std::printf("{\"event\":\"prefill_start\",\"since_process_start_seconds\":%.9g}\n", elapsed(process_start));
    std::fflush(stdout);
    const auto prefill_begin = Clock::now();
    // Additive prompt-phase split: prefill_process_seconds is the sum of the llama_process spans (the
    // submit calls below) and prefill_finish_seconds the finish_prompt span; the existing prefill_seconds
    // stays the composite prompt wall (process + finish + bookkeeping), do not add the parts on top of it.
    double prefill_process_seconds = 0;
    int64_t prefill_process_calls = 0;
    int rc = 0;
    if (prompt.size() <= cp.n_batch) {
        fill_batch(batch.get(), prompt.data(), (int) prompt.size(), 0);
        const auto process_begin = Clock::now();
        const int process_rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        prefill_process_seconds += elapsed(process_begin);
        ++prefill_process_calls;
        if (process_rc != 0) {
            std::fprintf(stderr, "strata generate: prefill failed\n");
            rc = 1;
        }
    } else {
        for (size_t at = 0; at < prompt.size() && rc == 0; at += cp.n_batch) {
            const int count = (int) std::min<size_t>(cp.n_batch, prompt.size() - at);
            fill_batch(batch.get(), prompt.data() + at, count, (int) at);
            const auto process_begin = Clock::now();
            const int process_rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            prefill_process_seconds += elapsed(process_begin);
            ++prefill_process_calls;
            if (process_rc != 0) {
                std::fprintf(stderr, "strata generate: prefill failed at %zu\n", at);
                rc = 1;
            }
        }
    }
    // The prompt -> decode boundary: refill the loaned cache slots and restore their residency (no-op when
    // the prompt workspace was owned).  Must run even after a failed prefill so the cache is left truthful;
    // a failed return is a REAL failure: no decode on a cache that still holds the prompt's workspace.
    const auto finish_begin = Clock::now();
    const bool loan_returned = core::qwen35_bridge_finish_prompt();
    const double prefill_finish_seconds = elapsed(finish_begin);
    if (!loan_returned && rc == 0) {
        std::fprintf(stderr, "strata generate: the prompt loan was not returned; stopping before decode\n");
        rc = 1;
    }
    // Graph observation boundary (default off): the request boundary never reads perf (a failed prefill has
    // no valid decode perf semantics); only the logger counters are reported.
    graph_stats_record("prompt_end", -1, nullptr);
    // prefill_seconds (and so TTFT and prefill tok/s) include the loan return: the refill is part of the
    // prompt phase and cannot be omitted from its cost.
    const double prefill_seconds = elapsed(prefill_begin);
    double ttft_seconds = -1;
    vram.sample();
    std::ofstream topk_stream;
    if (o.topk > 0 && !o.topk_file.empty()) {
        topk_stream.open(o.topk_file, std::ios::binary);
        if (!topk_stream) {
            std::fprintf(stderr, "strata generate: cannot open %s\n", o.topk_file.c_str());
            core::qwen35_bridge_shutdown();
            return 1;
        }
    }
    std::vector<llama_token> generated;
    double decode_seconds = 0;
    int decode_evaluations = 0;
    // Additive decode timing (observation only - no new synchronize, no ordering change):
    //  - decode_eval_seconds: sum over completed evals of [just before llama_process -> return of the next
    //    llama_get_logits_ith], so a span holds its process, the next round's VRAM sample, host gap and the
    //    sync wait.  It is a submission-to-sync wall span, NOT eval-only kernel time and not "fair" against
    //    a pure engine timer; the unfinished span of a failed process is discarded, and a NaN stop closes its
    //    span at that round's get_logits first (that eval did sync).
    //  - decode_observe_seconds: per-round NaN scan + sampler + topk dump + first-token/token JSON + flush,
    //    with additive sub-items: scan (NaN scan on the raw logits), sampler (llama_sampler_sample),
    //    topk (only while the dump is open; exactly 0 with topk disabled), piece_json_flush (round-0
    //    first-token output, piece/hex conversion, token JSON + fflush).  The sub-items partition
    //    decode_observe_seconds except the is_open check, the elapsed calls and the counters (bookkeeping).
    //  - decode_first_vram_seconds / decode_first_logits_wait_seconds: round 0's VRAM sample and get_logits
    //    wait (the prompt-tail wait under cache-off); both sit outside the eval spans and are split out.
    //  - decode_vram_seconds: all in-loop VRAM sample calls; sample 0 is the first-vram term, samples 1..31
    //    are inside the eval spans, so this sum overlaps them and must not be added to decode_eval_seconds.
    //    `decode_vram_samples` counts CALLS (successful or not); the old `vram_samples` counts cudaMemGetInfo
    //    SUCCESSES only, and `vram_sample_seconds_total` sums every process sample's wall time including the
    //    initial/post-decode ones outside the loop.
    // Bookkeeping identity: decode_seconds ~= decode_eval_seconds + decode_observe_seconds +
    // decode_first_vram_seconds + decode_first_logits_wait_seconds + loop bookkeeping (fill_batch, perf
    // reads, loop mechanics).  The original decode_seconds/decode_evaluations_per_s keep their composite
    // wall-clock meaning.
    double decode_eval_seconds = 0;
    int64_t decode_eval_spans = 0;
    double decode_observe_seconds = 0;
    int64_t decode_observe_rounds = 0;
    // Observe sub-item accumulators (see the bullet above).  A sub-item stays exactly 0 when its path never
    // runs (e.g. decode_topk_* with the topk dump closed); the round counts make partial/NaN rounds visible.
    double decode_scan_seconds = 0;
    int64_t decode_scan_rounds = 0;
    double decode_sampler_seconds = 0;
    int64_t decode_sampler_rounds = 0;
    double decode_topk_seconds = 0;
    int64_t decode_topk_rounds = 0;
    double decode_piece_json_flush_seconds = 0;
    int64_t decode_piece_json_flush_rounds = 0;
    // Sub-items INSIDE decode_piece_json_flush_seconds (its internal split, not additive to it): the
    // llama_token_to_piece call, the byte->hex conversion loop, every std::printf in the segment (round 0's
    // first-token line counts in that round), and every std::fflush(stdout) there.  Round counters count the
    // ROUNDS where the step ran (32 for a 32-output run; round 0 performs two writes/flushes but one round).
    double decode_piece_seconds = 0;
    int64_t decode_piece_rounds = 0;
    double decode_piece_hex_seconds = 0;
    int64_t decode_piece_hex_rounds = 0;
    double decode_token_write_seconds = 0;
    int64_t decode_token_write_rounds = 0;
    double decode_flush_seconds = 0;
    int64_t decode_flush_rounds = 0;
    double decode_first_vram_seconds = -1;
    double decode_first_logits_wait_seconds = -1;
    double decode_vram_seconds = 0;
    int64_t decode_vram_samples = 0;
    bool eval_span_open = false;
    Clock::time_point eval_span_begin{};
    // Engine perf via the public llama_perf_context (a pure read of the perf counters, no new synchronize):
    // first read right after round 0's get_logits (the prompt sync), last read after the loop's last
    // get_logits.  perf_get_data floors n_eval/n_p_eval at 1, so engine_perf_n_eval_delta is a delta of
    // floored counters (it can undercount by 1 while the first read has t_eval_ms == 0); t_eval_ms is raw.
    // This engine delta uses a different clock than the decode_eval wall span - never subtract the two to
    // estimate an observation-free speed.
    bool have_perf_first = false;
    llama_perf_context_data perf_first = {};
    llama_perf_context_data perf_last = {};
    if (rc == 0) {
        const auto decode_begin = Clock::now();
        const int64_t n_vocab = llama_vocab_n_tokens(vocab);
        for (int64_t i = 0; i < o.max_new; ++i) {
            const double vram_seconds = vram.sample();   // per-token sample: bounded spot coverage of the peak, not a continuous trace
            decode_vram_seconds += vram_seconds;
            ++decode_vram_samples;
            if (i == 0) decode_first_vram_seconds = vram_seconds;
            const auto logits_begin = Clock::now();
            const float* logits = llama_get_logits_ith(ctx.get(), -1);
            const double logits_seconds = elapsed(logits_begin);
            if (i == 0) decode_first_logits_wait_seconds = logits_seconds;
            if (eval_span_open) {
                decode_eval_seconds += elapsed(eval_span_begin);
                ++decode_eval_spans;
                eval_span_open = false;
            }
            if (!have_perf_first) {
                perf_first = llama_perf_context(ctx.get());
                have_perf_first = true;
            }
            const auto observe_begin = Clock::now();
            const auto scan_begin = Clock::now();
            const bool scan_found_nan = logits != nullptr && logits_have_nan(logits, n_vocab);
            decode_scan_seconds += elapsed(scan_begin);
            ++decode_scan_rounds;
            if (scan_found_nan) {
                decode_observe_seconds += elapsed(observe_begin);
                ++decode_observe_rounds;
                std::fprintf(stderr, "strata generate: NaN in the logits at output %lld; stopping\n",
                             (long long) i);
                rc = 1;
                break;
            }
            const auto sampler_begin = Clock::now();
            const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            decode_sampler_seconds += elapsed(sampler_begin);
            ++decode_sampler_rounds;
            if (topk_stream.is_open()) {
                const auto topk_begin = Clock::now();
                if (logits != nullptr) {
                    std::vector<int32_t> idx((size_t) n_vocab);
                    for (int64_t v = 0; v < n_vocab; ++v) idx[(size_t) v] = (int32_t) v;
                    const int64_t top = std::min<int64_t>(o.topk, n_vocab);
                    std::partial_sort(idx.begin(), idx.begin() + (size_t) top, idx.end(),
                                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                    topk_stream << "{\"index\":" << i << ",\"sampled\":" << token << ",\"top\":[";
                    for (int64_t t = 0; t < top; ++t)
                        topk_stream << (t ? "," : "") << "[" << idx[(size_t) t] << ","
                                    << std::setprecision(9) << logits[idx[(size_t) t]] << "]";
                    topk_stream << "]}\n";
                }
                decode_topk_seconds += elapsed(topk_begin);
                ++decode_topk_rounds;
            }
            const auto piece_begin = Clock::now();
            double token_write_seconds = 0;
            double token_flush_seconds = 0;
            if (i == 0) {
                ttft_seconds = elapsed(prefill_begin);
                const auto write_begin = Clock::now();
                std::printf("{\"event\":\"first_token\",\"id\":%d,\"since_prefill_start_seconds\":%.9g,"
                            "\"since_process_start_seconds\":%.9g}\n", token, ttft_seconds,
                            elapsed(process_start));
                token_write_seconds += elapsed(write_begin);
                const auto flush_begin = Clock::now();
                std::fflush(stdout);
                token_flush_seconds += elapsed(flush_begin);
            }
            generated.push_back(token);
            char piece[256];
            const auto piece_call_begin = Clock::now();
            const int size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            decode_piece_seconds += elapsed(piece_call_begin);
            ++decode_piece_rounds;
            std::string hex;
            const auto hex_begin = Clock::now();
            for (int j = 0; j < (size > 0 ? size : 0); ++j) {
                char byte[3];
                std::snprintf(byte, sizeof(byte), "%02x", (unsigned char) piece[j]);
                hex += byte;
            }
            decode_piece_hex_seconds += elapsed(hex_begin);
            ++decode_piece_hex_rounds;
            {
                const auto write_begin = Clock::now();
                std::printf("{\"event\":\"token\",\"index\":%lld,\"id\":%d,\"teacher\":%s,\"piece_hex\":%s}\n",
                            (long long) i, token, teacher.empty() ? "false" : "true", json_quote(hex).c_str());
                token_write_seconds += elapsed(write_begin);
            }
            {
                const auto flush_begin = Clock::now();
                std::fflush(stdout);
                token_flush_seconds += elapsed(flush_begin);
            }
            decode_token_write_seconds += token_write_seconds;
            ++decode_token_write_rounds;
            decode_flush_seconds += token_flush_seconds;
            ++decode_flush_rounds;
            decode_piece_json_flush_seconds += elapsed(piece_begin);
            ++decode_piece_json_flush_rounds;
            decode_observe_seconds += elapsed(observe_begin);
            ++decode_observe_rounds;
            // Graph observation boundary (default off): one record per generated output; perf is read only
            // after the same round-0 point the runner itself validates (have_perf_first).
            graph_stats_record("output", (long long) i, have_perf_first ? ctx.get() : nullptr);
            if (i + 1 < o.max_new) {
                const llama_token fed = teacher.empty() ? token : teacher[(size_t) i];
                fill_batch(batch.get(), &fed, 1, (int) (prompt.size() + i));
                eval_span_begin = Clock::now();
                const int process_rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (process_rc != 0) {
                    std::fprintf(stderr, "strata generate: decode failed at %lld\n", (long long) i);
                    rc = 1;
                    break;
                }
                eval_span_open = true;
                ++decode_evaluations;
            }
        }
        decode_seconds = elapsed(decode_begin);
        perf_last = llama_perf_context(ctx.get());
    }
    vram.sample();
    core::Qwen35BridgeStats st;
    core::qwen35_bridge_stats(st);
    const double prefill_tps = prefill_seconds > 0 ? (double) prompt.size() / prefill_seconds : 0.0;
    const double decode_eps = decode_seconds > 0 ? (double) decode_evaluations / decode_seconds : 0.0;
    // Additive rate: completed eval spans per decode_eval_seconds (submission-to-sync wall), 0 when there
    // is no completed span (max_new 0/1, failed first process), so it never divides by zero.
    const double decode_eval_per_s = (decode_eval_spans > 0 && decode_eval_seconds > 0)
                                             ? (double) decode_eval_spans / decode_eval_seconds : 0.0;
    // Engine-perf deltas from the public llama_perf_context: first read after round 0's get_logits (the
    // prompt sync), last read after the loop's last get_logits.  perf_get_data floors n_eval/n_p_eval at 1,
    // so engine_perf_n_eval_delta is a delta of floored counters (can undercount by 1 while the first read
    // has t_eval_ms == 0); t_eval_ms itself is raw.  These differ in clock and boundary from the decode
    // wall spans above - do not subtract them to estimate an observation-free speed.
    double engine_t_eval_ms_first = -1, engine_t_eval_ms_last = -1, engine_t_eval_ms_delta = -1;
    double engine_t_p_eval_ms_first = -1;
    long long engine_n_eval_first = -1, engine_n_eval_last = -1, engine_n_eval_delta = -1;
    long long engine_n_p_eval_first = -1;
    if (have_perf_first) {
        engine_t_eval_ms_first = perf_first.t_eval_ms;
        engine_t_eval_ms_last = perf_last.t_eval_ms;
        engine_t_eval_ms_delta = perf_last.t_eval_ms - perf_first.t_eval_ms;
        engine_t_p_eval_ms_first = perf_first.t_p_eval_ms;
        engine_n_eval_first = (long long) perf_first.n_eval;
        engine_n_eval_last = (long long) perf_last.n_eval;
        engine_n_eval_delta = (long long) perf_last.n_eval - (long long) perf_first.n_eval;
        engine_n_p_eval_first = (long long) perf_first.n_p_eval;
    }
    std::printf("{\"event\":\"summary\",\"exit\":%d,\"prompt_tokens\":%zu,\"output_tokens\":%zu,"
                "\"decode_evaluations\":%d,\"prefill_seconds\":%.9g,\"decode_seconds\":%.9g,"
                "\"ttft_seconds\":%.9g,\"prefill_tokens_per_s\":%.6g,\"decode_evaluations_per_s\":%.6g,"
                "\"strata_jobs\":%llu,\"strata_seconds\":%.6g,\"decode_layers\":%llu,\"decode_hits\":%llu,"
                "\"decode_misses\":%llu,\"prefill_layers\":%llu,\"prefill_entries\":%llu,"
                "\"cache_slots\":%llu,\"cache_bytes\":%llu,\"cache_mib\":%.1f,\"cache_payload_mib\":%.1f,"
                "\"cache_resident\":%llu,"
                "\"hit_overlap\":%lld,\"hit_late\":%lld,\"hit_inflight\":%lld,\"all_hit_layers\":%lld,"
                "\"prefill_gpu_layers\":%lld,\"prefill_resident\":%lld,\"prefill_streamed\":%lld,"
                "\"prefill_product_calls\":%lld,\"prefill_stream_mib\":%.2f,\"prefill_workspace_mib\":%.2f,"
                "\"prefill_ring_mib\":%.2f,"
                "\"prefill_loan_slots\":%lld,\"prefill_loan_mib\":%.2f,\"prefill_owned_mib\":%.2f,"
                "\"prefill_loan_restored\":%lld,\"prefill_loan_refill_ms\":%.1f,\"prefill_loan_why\":%s,"
                "\"tier_ms\":{\"ids\":%.1f,\"launch\":%.1f,\"misses\":%.1f,\"combine\":%.1f,\"admit\":%.1f},"
                "\"vram_samples\":%lld,\"peak_device_vram_mib\":%.1f,\"device_total_mib\":%.1f,"
                "\"prefill_process_seconds\":%.9g,\"prefill_process_calls\":%lld,\"prefill_finish_seconds\":%.9g,"
                "\"decode_eval_seconds\":%.9g,\"decode_eval_spans\":%lld,\"decode_eval_per_s\":%.6g,"
                "\"decode_observe_seconds\":%.9g,\"decode_observe_rounds\":%lld,"
                "\"decode_first_vram_seconds\":%.9g,\"decode_first_logits_wait_seconds\":%.9g,"
                "\"decode_vram_seconds\":%.9g,\"decode_vram_samples\":%lld,\"vram_sample_seconds_total\":%.9g,"
                "\"engine_perf_t_eval_ms_first\":%.6g,\"engine_perf_t_eval_ms_last\":%.6g,"
                "\"engine_perf_t_eval_ms_delta\":%.6g,\"engine_perf_n_eval_first\":%lld,"
                "\"engine_perf_n_eval_last\":%lld,\"engine_perf_n_eval_delta\":%lld,"
                "\"engine_perf_t_p_eval_ms_first\":%.6g,\"engine_perf_n_p_eval_first\":%lld,"
                "\"shared_overlap_marker_layers\":%llu,\"shared_overlap_join_layers\":%llu,"
                "\"decode_scan_seconds\":%.9g,\"decode_scan_rounds\":%lld,"
                "\"decode_sampler_seconds\":%.9g,\"decode_sampler_rounds\":%lld,"
                "\"decode_topk_seconds\":%.9g,\"decode_topk_rounds\":%lld,"
                "\"decode_piece_json_flush_seconds\":%.9g,\"decode_piece_json_flush_rounds\":%lld,"
                "\"decode_piece_seconds\":%.9g,\"decode_piece_rounds\":%lld,"
                "\"decode_piece_hex_seconds\":%.9g,\"decode_piece_hex_rounds\":%lld,"
                "\"decode_token_write_seconds\":%.9g,\"decode_token_write_rounds\":%lld,"
                "\"decode_flush_seconds\":%.9g,\"decode_flush_rounds\":%lld}\n",
            rc, prompt.size(), generated.size(), decode_evaluations, prefill_seconds, decode_seconds,
            ttft_seconds, prefill_tps, decode_eps, (unsigned long long) st.jobs, st.expert_seconds,
            (unsigned long long) st.decode_layers, (unsigned long long) st.decode_hits,
            (unsigned long long) st.decode_misses, (unsigned long long) st.prefill_layers,
            (unsigned long long) st.prefill_entries, (unsigned long long) st.cache_slots,
            (unsigned long long) st.cache_bytes, (double) st.cache_bytes / 1048576.0,
            (double) st.cache_payload_bytes / 1048576.0,
            (unsigned long long) st.cache_resident, (long long) st.hit_overlap, (long long) st.hit_late,
            (long long) st.hit_inflight, (long long) st.all_hit, (long long) st.prefill_layers_gpu,
            (long long) st.prefill_resident, (long long) st.prefill_streamed,
            (long long) st.prefill_product_calls, (double) st.prefill_stream_bytes / 1048576.0,
            (double) st.prefill_workspace_bytes / 1048576.0, (double) st.prefill_ring_bytes / 1048576.0,
            (long long) st.prefill_loan_slots, (double) st.prefill_loan_bytes / 1048576.0,
            (double) st.prefill_owned_bytes / 1048576.0, (long long) st.prefill_loan_restored,
            st.prefill_loan_refill_ms, json_quote(st.prefill_loan_why).c_str(),
            st.ms_ids, st.ms_launch, st.ms_misses, st.ms_combine, st.ms_queue, (long long) vram.samples,
            vram.mib(), (double) vram.total / 1048576.0,
            prefill_process_seconds, (long long) prefill_process_calls, prefill_finish_seconds,
            decode_eval_seconds, (long long) decode_eval_spans, decode_eval_per_s,
            decode_observe_seconds, (long long) decode_observe_rounds,
            decode_first_vram_seconds, decode_first_logits_wait_seconds,
            decode_vram_seconds, (long long) decode_vram_samples, vram.sample_seconds,
            engine_t_eval_ms_first, engine_t_eval_ms_last, engine_t_eval_ms_delta,
            engine_n_eval_first, engine_n_eval_last, engine_n_eval_delta,
            engine_t_p_eval_ms_first, engine_n_p_eval_first,
            (unsigned long long) st.shared_overlap_marker_layers,
            (unsigned long long) st.shared_overlap_join_layers,
            decode_scan_seconds, (long long) decode_scan_rounds,
            decode_sampler_seconds, (long long) decode_sampler_rounds,
            decode_topk_seconds, (long long) decode_topk_rounds,
            decode_piece_json_flush_seconds, (long long) decode_piece_json_flush_rounds,
            decode_piece_seconds, (long long) decode_piece_rounds,
            decode_piece_hex_seconds, (long long) decode_piece_hex_rounds,
            decode_token_write_seconds, (long long) decode_token_write_rounds,
            decode_flush_seconds, (long long) decode_flush_rounds);
    std::fflush(stdout);
    core::qwen35_report_decode_seam();
    core::qwen35_bridge_shutdown();
    llama_backend_free();
    return rc;
}

namespace {

std::string meta_get(const std::string& text, const std::string& key, const std::string& fallback) {
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t eol = text.find('\n', pos);
        const std::string line = text.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? text.size() : eol + 1;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        if (line.compare(0, sp, key) == 0) return line.substr(sp + 1);
    }
    return fallback;
}

int64_t meta_i64(const std::string& text, const std::string& key, int64_t fallback) {
    const std::string v = meta_get(text, key, "");
    if (v.empty()) return fallback;
    return std::strtoll(v.c_str(), nullptr, 10);
}

bool read_blob(const std::string& path, void* dst, size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read((char*) dst, (std::streamsize) bytes);
    return (size_t) in.gcount() == bytes;
}

}  // namespace

int qwen35_layer_replay(const Qwen35RunOptions& o, const std::string& capture_dir, bool mixed) {
    const auto fail = [](const std::string& what) {
        std::fprintf(stderr, "strata generate: layer replay: %s\n", what.c_str());
        return 1;
    };
    std::ifstream mf(capture_dir + "/meta.txt");
    if (!mf) return fail("cannot open " + capture_dir + "/meta.txt");
    const std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const int64_t layer = meta_i64(meta, "layer", -1);
    const int64_t T = meta_i64(meta, "T", -1);
    const int64_t K = meta_i64(meta, "K", -1);
    const int64_t H = meta_i64(meta, "H", -1);
    const int64_t FF = meta_i64(meta, "FF", -1);
    const int64_t n_layers = meta_i64(meta, "n_layers", -1);
    const int64_t n_expert = meta_i64(meta, "n_expert", -1);
    const int64_t blob_bytes = meta_i64(meta, "blob_bytes", -1);
    const bool perturbed = meta_i64(meta, "perturbed", 0) != 0;
    if (layer != 16 || T < 2 || T > 4096 || K < 1 || K > 16 || H < 1 || FF < 1 || n_layers < 1 || n_expert < 1 ||
        blob_bytes < 1) {
        return fail("the capture meta is incomplete or out of range");
    }
    std::vector<float> x((size_t) T * H), w((size_t) T * K), ref((size_t) T * H);
    std::vector<int32_t> ids((size_t) T * K), mask((size_t) n_expert, -1);
    if (!read_blob(capture_dir + "/x.f32", x.data(), x.size() * sizeof(float)) ||
        !read_blob(capture_dir + "/weights.f32", w.data(), w.size() * sizeof(float)) ||
        !read_blob(capture_dir + "/out.f32", ref.data(), ref.size() * sizeof(float)) ||
        !read_blob(capture_dir + "/ids.i32", ids.data(), ids.size() * sizeof(int32_t)) ||
        !read_blob(capture_dir + "/mask.i32", mask.data(), mask.size() * sizeof(int32_t))) {
        return fail("cannot read the capture arrays with the expected sizes");
    }
    std::string err;
    namespace kc = strata::kernels::cpu;
    if (!kc::expert_layout_load(o.pack_dir, n_layers, n_expert, H, FF, err)) return fail("layout: " + err);
    const kc::ExpertLayout& lay = kc::expert_layout();
    if (layer >= (int64_t) lay.fmt.size()) return fail("the capture layer is outside the pack");
    if ((int64_t) lay.blob_bytes(layer) != blob_bytes) return fail("the capture blob size does not match the pack");
    core::FileExpertSource src;
    src.set_gguf(o.model);
    if (!src.open(o.pack_dir, n_layers, n_expert, err)) return fail("source: " + err);
    std::unique_ptr<kc::ExpertPool> pool = std::make_unique<kc::ExpertPool>(1, true, true);
    core::ExpertDispatch d{};
    d.pool = pool.get();
    d.src = &src;
    d.n_expert = n_expert;
    d.n_embd = H;
    d.n_ff = FF;
    d.native_k = K;
    if (!core::expert_native_cache_open(d, 256, std::string(), err)) return fail("cache: " + err);

    // Loading/fill is listed separately: it is NOT part of the MoE call wall below.
    const auto fill_t0 = Clock::now();
    std::vector<uint8_t> host_blob((size_t) blob_bytes);
    int64_t installed = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        if (mixed && mask[(size_t) e] < 0) continue;
        const int32_t slot = d.cache->admit(layer, e);
        if (slot < 0) return fail("the cache refused a slot");
        if (!src.copy_blob(layer, e, host_blob.data())) return fail("a source blob copy failed");
        if (!d.cache->fill_slot_queued(slot, host_blob.data(), err, blob_bytes)) return fail("fill: " + err);
        ++installed;
    }
    if (!d.cache->sync_queued(err)) return fail("fill sync: " + err);
    // Milliseconds (the old label printed seconds): the GGUF blob reads + queued fills + fill sync, host wall.
    // This loading/fill wall is NOT part of the MoE call wall below.
    const double fill_ms = elapsed(fill_t0) * 1000.0;

    void* x_dev = nullptr; void* ids_dev = nullptr; void* w_dev = nullptr; void* out_dev = nullptr;
    bool ok = cudaMalloc(&x_dev, (size_t) T * H * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&ids_dev, (size_t) T * K * sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc(&w_dev, (size_t) T * K * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&out_dev, (size_t) T * H * sizeof(float)) == cudaSuccess &&
              cudaMemcpy(x_dev, x.data(), (size_t) T * H * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(ids_dev, ids.data(), (size_t) T * K * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(w_dev, w.data(), (size_t) T * K * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    const auto cleanup = [&]() {
        core::qwen_prefill_close(d);
        core::expert_native_close(d);
        if (x_dev) cudaFree(x_dev);
        if (ids_dev) cudaFree(ids_dev);
        if (w_dev) cudaFree(w_dev);
        if (out_dev) cudaFree(out_dev);
        src.close();
    };
    if (!ok) { cleanup(); return fail("cannot allocate/copy the replay's device buffers"); }

    // THE SAME original call: same interface, same order/group16, unchanged math.
    std::string err2;
    const bool first_alloc_inside = d.prefill_state == nullptr;   // the workspace's first alloc happens inside
    const auto moe_t0 = Clock::now();
    const bool call_ok = core::qwen_prefill_experts(d, layer, T, x_dev, H * (int64_t) sizeof(float), ids_dev,
                                                    K * (int64_t) sizeof(int32_t), w_dev, (int64_t) sizeof(float),
                                                    K * (int64_t) sizeof(int32_t), out_dev, nullptr, err2);
    const cudaError_t sy = cudaDeviceSynchronize();
    // Milliseconds: the qwen_prefill_experts call plus the device sync (inference wall).  It INCLUDES the
    // workspace's first allocation (alloc_for runs inside the call) - first_alloc_inside reports that; it does
    // NOT include the GGUF loading/cache fill above.
    const double moe_ms = elapsed(moe_t0) * 1000.0;
    std::vector<float> got((size_t) T * H, 0.0f);
    const bool back_ok = cudaMemcpy(got.data(), out_dev, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    int64_t mism = 0, nf = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) ++nf;
        if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) ++mism;
    }
    // Task 65 diagnostic (default off): exact STRATA_QWEN35_LAYER_REPLAY_OUTPUT=<absolute path> writes the raw
    // got buffer (T*H f32, row-major [T, H]) once for external error statistics.  Unset or empty = no IO.
    // The existing ref_bitwise/mism and exit rules are unchanged; a requested dump is failed explicitly when
    // the buffer did not come back or the write is incomplete, and the dump is still written when values are
    // non-finite (that is exactly what the raw data must show).
    bool dump_ok = true;
    if (const char* dump_path = std::getenv("STRATA_QWEN35_LAYER_REPLAY_OUTPUT"); dump_path != nullptr && dump_path[0] != '\0') {
        dump_ok = false;
        if (!back_ok) {
            std::fprintf(stderr, "strata qwen35: layer replay: error: the got buffer did not come back; no dump "
                                 "written to \"%s\"\n", dump_path);
        } else if (std::FILE* df = std::fopen(dump_path, "wb"); df == nullptr) {
            std::fprintf(stderr, "strata qwen35: layer replay: error: cannot open the got dump \"%s\"\n", dump_path);
        } else {
            const bool wr = std::fwrite(got.data(), sizeof(float), got.size(), df) == got.size();
            const bool cl = std::fclose(df) == 0;
            if (wr && cl) {
                dump_ok = true;
                std::fprintf(stderr, "strata qwen35: layer replay got dump: path \"%s\" shape [%lld,%lld] f32 bytes %lld\n",
                             dump_path, (long long) T, (long long) H, (long long) (got.size() * sizeof(float)));
            } else {
                std::fprintf(stderr, "strata qwen35: layer replay: error: the got dump \"%s\" was not written completely\n",
                             dump_path);
            }
        }
    }
    std::vector<uint8_t> seen((size_t) n_expert, 0);
    int64_t distinct = 0;
    for (int32_t v : ids)
        if (v >= 0 && v < n_expert && !seen[(size_t) v]) { seen[(size_t) v] = 1; ++distinct; }
    int64_t cold = 0;
    for (int64_t e = 0; e < n_expert; ++e)
        if (seen[(size_t) e] && (d.cache == nullptr || d.cache->slot_of(layer, e) < 0)) ++cold;
    const int64_t resident = distinct - cold;
    const int64_t groups = (distinct + 15) / 16;
    const std::string loan_why = d.native_stats.prefill_loan_why;
    const long long owned = (long long) d.native_stats.prefill_owned_bytes;
    const long long borrowed = (long long) d.native_stats.prefill_loan_bytes;
    std::fprintf(stderr, "strata qwen35: layer replay: arm %s layer %lld T %lld K %lld H %lld installed %lld "
                         "resident %lld cold %lld distinct %lld groups %lld rows %lld ref_bitwise %d mism %lld "
                         "out_nonfinite %lld call_ok %d cuda %d cache_slots %lld cache_bytes %lld owned %lld "
                         "borrowed %lld loan_why \"%s\" fill_ms %.6f moe_ms %.6f first_alloc_inside %d perturbed %d\n",
                 mixed ? "capturedmixed" : "allresident", (long long) layer, (long long) T, (long long) K,
                 (long long) H, (long long) installed, (long long) resident, (long long) cold,
                 (long long) distinct, (long long) groups, (long long) (T * K), mism == 0 ? 1 : 0,
                 (long long) mism, (long long) nf, call_ok ? 1 : 0, (int) sy, (long long) d.cache->slots(),
                 (long long) d.cache->bytes(), owned, borrowed, loan_why.empty() ? "borrowed" : loan_why.c_str(),
                 fill_ms, moe_ms, first_alloc_inside ? 1 : 0, perturbed ? 1 : 0);
    cleanup();
    if (!call_ok) {
        std::fprintf(stderr, "strata generate: layer replay: the expert call failed: %s\n", err2.c_str());
        return 1;
    }
    if (sy != cudaSuccess || !back_ok) return fail("the replay's device work did not complete");
    if (!dump_ok) return fail("the requested got dump was not written");
    if (mism != 0 || nf != 0) return fail("the replay output is not bitwise equal to the capture reference");
    return 0;
}

}  // namespace strata::program
