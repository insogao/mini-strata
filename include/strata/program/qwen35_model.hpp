// include/strata/program/qwen35_model.hpp - the original `strata generate` entry's Qwen35 text path.
//
// The model graph is the patched llama.cpp checkout (`LLM_ARCH_QWEN35MOE`); the routed experts go through
// `strata::core::qwen35_bridge` into the shared expert service.  This header is only the runner's options.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {

struct Qwen35RunOptions {
    std::string model;            ///< the qwen35moe GGUF
    std::string pack_dir;         ///< directory holding native_experts.txt
    std::vector<int64_t> tokens;  ///< the prompt, already tokenized (fixed test IDs)
    std::string prompt_text;      ///< alternative prompt: tokenized with the model's own vocab (calibration)
    std::string teacher_file;     ///< optional fixed prefix: fed instead of the sampled token (perf attribution)
    int64_t max_new = 32;
    int64_t max_context = 4096;
    int64_t ngl = 99;
    int64_t threads = 8;
    int64_t ubatch = 128;
    int64_t cache_slots = 0;      ///< GPU expert cache slots (0 = off)
    std::string profile;          ///< optional expert profile for the cache fill
    std::string routing_log;      ///< optional diagnostic routing log
    int64_t topk = 0;             ///< dump the top-N logits of every sampled position (0 = off)
    std::string topk_file;        ///< where the top-k dump goes (JSONL)
    std::string cuda_backend;     ///< path to the ggml-cuda dynamic backend (built in when empty)
    int pool_workers = 0;
    bool dump_routing = false;
};

/// Runs prefill + greedy decode and prints the runner events (prompt_tokens / token / summary).  Exit code 0
/// only when the model ran and the bridge's service was actually used.
int qwen35_run(const Qwen35RunOptions& o);

/// Task 52 diagnostic replay (default off; STRATA_QWEN35_LAYER_REPLAY=<capture dir>): replays one captured
/// prompt layer through the SAME `qwen_prefill_experts` interface with a standalone 256-slot expert cache.
/// No llama context and no dense model are loaded: only the pack layout and the GGUF's expert slices are
/// mapped.  `mixed` installs exactly the captured resident set; otherwise all 256 layer experts are resident.
/// Prints one `strata qwen35: layer replay` summary line; 0 only when the replay's output is bitwise equal to
/// the captured reference and every guard passed.
int qwen35_layer_replay(const Qwen35RunOptions& o, const std::string& capture_dir, bool mixed);

}  // namespace strata::program
