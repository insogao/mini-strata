// src/../include/strata/core/qwen35_mtp_txn.hpp - task 257/259: default-OFF greedy K=1 MTP transaction engine.
//
// A minimal, real, single-request diagnostic chain for the vendor MTP block: draft round (real seed token +
// deep-copied previous target hidden) -> target TOKEN-ONLY verify (all rows) -> greedy contiguous prefix
// a in [0,1] -> target KV rollback / rs_idx pending -> draft independent-KV sync of the accepted prefix
// (accepted token + deep-copied verify row0 hidden) with verify row1 as the next pending hidden.  Everything
// here is gated by STRATA_QWEN35_MTP_TXN=1 (exact value) and defaults OFF; the OFF path never calls into this
// module.
//
// Task 259 emission contract: this engine is the ONE output owner for a txn round.  It emits every visible
// greedy token exactly once in order - the prompt-sampled seed, the accepted draft, the accepted bonus /
// rejected corrective pending, and the context-tail serial emit - via qwen35_mtp_txn_emit events and the
// returned emitted vector.  max_new is the visible-token budget (rounds counted separately); EOS stops
// without emitting the EOG token or replaying it; pos >= n_ctx fails before any decode.
//
// Scope honesty: this is the transaction ENGINE and diagnostic entry only.  No accept-rate, speed or business
// quality claim comes from it; K=1 greedy only (K>1 is rejected fail-closed); the target is never fed draft
// hidden states and the draft is never fed fake/teacher-forced tokens.
#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Exact STRATA_QWEN35_MTP_TXN=1 gate (default off).  Distinct from the bootstrap-only STRATA_QWEN35_MTP_DRAFT.
bool qwen35_mtp_txn_enabled();

/// Requested K (STRATA_QWEN35_MTP_TXN_K, default 1).  The engine implements greedy K=1 only.
int64_t qwen35_mtp_txn_k();

/// Pure greedy decision core (no llama calls, unit-testable): for K=1 with verify rows (t0 = target greedy
/// after the pending token, t1 = target greedy after the draft token), a = (t0 == draft_id) ? 1 : 0.
///   accept (a=1): emit = draft_id, next_pending = t1, next_pos_delta = 2, rollback_rows = 0
///   reject (a=0): emit = t0,       next_pending = t0, next_pos_delta = 1, rollback_rows = 1
void qwen35_mtp_txn_decide(llama_token draft_id, llama_token t0, llama_token t1, llama_token* emit,
                           llama_token* next_pending, int64_t* next_pos_delta, int64_t* rollback_rows);

/// Task 264: per-request real-clock observation returned by the engine.  Every `_s` field is host
/// steady_clock seconds measured INSIDE the engine call - `engine_wall_s` counts from the engine entry to
/// the final stop decision, the per-emit vectors from the entry to the token's write+flush.  These are
/// engine-relative instants: they are NOT the request E2E and the caller composes them with its own
/// request anchor (engine entry instant + these values) before presenting any request-level number.
/// Vectors have one entry per actually emitted visible token; values are -1.0 only when the event never
/// happened.  Populated only on success (ok = true); a fail-closed run leaves ok = false.
struct Qwen35MtpTxnTiming {
    bool ok = false;
    double engine_wall_s = -1.0;        // entry -> the stop decision (before the done event's own write)
    double first_ready_s = -1.0;        // entry -> first visible token decided (before its write)
    double first_flush_s = -1.0;        // entry -> first visible token written and flushed (real TTFT side)
    std::vector<double> emit_ready_s;   // per emitted token: entry -> decided (before its write)
    std::vector<double> emit_flush_s;   // per emitted token: entry -> written AND flushed
    int64_t rounds = 0;
    int64_t emitted = 0;
    int64_t eos_count = 0;
    int64_t end_pos = 0;
    long long eog_id = -1;              // the EOG token that stopped generation (-1 when none)
    std::string stop;                   // the done event's stop reason
    // Task 267 default-OFF stage cost accounting (STRATA_QWEN35_MTP_COST_SPLIT=1).  Each phase value is
    // HOST wall time (first steady_clock read before the phase -> first read after it): it contains the
    // host submission work plus any wait already caused by a displaced readback - it is NOT pure GPU
    // kernel time and overlapping async windows must never be summed as if they were.  The
    // optional cost_sync (STRATA_QWEN35_MTP_COST_SYNC=1) adds one public llama_synchronize at each phase
    // end (diagnostic only); cost_sync_calls/cost_sync_wait_s carry that extra wait separately and the
    // synchronized diag is NOT a product speed, while the plain control arm (0) is untouched.
    bool cost_split_on = false;
    bool cost_sync_on = false;
    double ph_initdraftctx_ms = 0.0;      // draft MTP ctx create + embeddings-nextn enable (host submit)
    double ph_draft_ms = 0.0;             // draft decode + logits + argmax
    double ph_verify_ms = 0.0;            // target T2 verify (all rows) + both row logits/argmax
    double ph_hcopy_ms = 0.0;             // nextn hidden rows: copy + the row finiteness scan
    double ph_replay_ms = 0.0;            // accepted-prefix draft KV sync decode (the replay)
    double ph_rollback_ms = 0.0;          // target seq_rm rollback
    double ph_emit_ms = 0.0;              // every emit/round/done printf + fflush (host writes)
    double ph_unattributed_ms = 0.0;      // engine_wall - the phases above (loop bookkeeping + stop check)
    int64_t cost_sync_calls = 0;
    double cost_sync_wait_s = 0.0;
};

/// Runs the bounded K=1 greedy transaction diagnostic on an already-prefilled target context:
///   model/tgt  : the loaded model and the target context (created with n_outputs_max >= 2, n_rs_seq >= 1,
///                embeddings_nextn enabled) after the prompt prefill, with the request sequence at seq 0.
///   first_pos  : position of the first pending token (the token the prompt logits sampled, not yet decoded).
///   first_token: that pending token (real sampled seed).
///   max_new    : visible-token budget (the caller's max-new); rounds/EOS are accounted separately.
///   emitted_out: optional; on success receives every visible greedy token exactly once, in order (seed,
///                accepted draft, accepted bonus / rejected corrective pending, ctx-tail serial emit).
///   timing_out : optional; on success receives the per-request real-clock observation above.
/// Returns 0 on success, non-zero with *err on ANY fail-closed condition (unmet n_rs_seq capability, K>1,
/// null row/logits, rollback refusal, decode rc != 0, pos >= n_ctx).  Never falls back to a fake draft or
/// teacher forcing.  An EOG pending seed/draft/bonus/corrective/tail token stops generation without being
/// emitted and is never decoded or replayed again.
int qwen35_mtp_txn_run(llama_model* model, llama_context* tgt, const llama_vocab* vocab, int64_t first_pos,
                       llama_token first_token, int64_t max_new, std::vector<llama_token>* emitted_out,
                       Qwen35MtpTxnTiming* timing_out, std::string* err);

}  // namespace strata::core
