// src/core/qwen35_mtp_txn.cpp - task 257/259: greedy K=1 MTP transaction engine (default OFF diagnostic).
//
// Task 268 candidate (STRATA_QWEN35_MTP_SERIAL_VERIFY=1, default OFF): the target verify of a round becomes a
// serial of token-only T1 decodes instead of one T2 batch - T1#1 decodes the pending token alone, its t0/h0
// are read and copied immediately; only when draft_id == t0 (and the draft token is legal, i.e. not EOG) a
// second T1#2 decodes the draft token alone and its t1/h1 are copied immediately.  On reject no speculative
// target row was ever decoded, so the target sequence is already clean (nothing to seq_rm); the corrective is
// still t0/h0.  On accept the real draft token + hidden replay and every seed/bonus/EOS/max_new/context-tail
// rule are unchanged.  OFF = the previous T2 path, unchanged.
//
// Task 259 fixes (CODEX_REVIEW_256_257_258):
//   1. The accepted-prefix draft replay is token+hidden: the accepted token at pos+1 is decoded into the
//      draft context together with a deep copy of target verify row0 (hidden after the pending token).
//      Token-only replay would not provide h (the vendor graph_mtp always allocates an h input and its
//      set_input fills it only when a row carries embeddings).  The next pending hidden is verify row1.
//      Only the real accepted token is ever replayed; no future/fabricated draft token is.
//   2. One output owner: this engine emits every visible greedy token exactly once, in order - the
//      prompt-sampled seed, the accepted draft, the accepted bonus / rejected corrective pending, and the
//      context-tail serial emit.  The caller no longer emits anything for a txn round.
//   3. max_new is the visible-token budget; rounds/EOS are accounted separately.  EOS stops without
//      emitting the EOG token and without replaying it; pos >= n_ctx fails before any decode; only one
//      visible-token budget remaining does one legal final serial/round-start emit and stops.
#include "strata/core/qwen35_mtp_txn.hpp"

#include "llama-ext.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace strata::core {
namespace {

std::vector<float> copy_nextn_row(llama_context* tgt, int32_t row, bool* ok) {
    *ok = false;
    const int64_t n_embd_out = (int64_t) llama_model_n_embd_out(llama_get_model(tgt));
    const float* src = llama_get_embeddings_nextn_ith(tgt, row);
    if (src == nullptr || n_embd_out <= 0) return {};
    std::vector<float> out((size_t) n_embd_out);
    std::memcpy(out.data(), src, (size_t) n_embd_out * sizeof(float));
    for (float v : out) {
        if (!std::isfinite(v)) return {};
    }
    *ok = true;
    return out;
}

llama_token greedy_argmax(const float* logits, int64_t n_vocab, bool* ok) {
    *ok = false;
    if (logits == nullptr || n_vocab <= 0) return -1;
    int64_t best = -1;
    double best_v = 0.0;
    for (int64_t i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) return -1;
        if (best < 0 || logits[i] > best_v) { best = i; best_v = logits[i]; }
    }
    *ok = true;
    return (llama_token) best;
}

bool decode_rows(llama_context* ctx, const llama_token* ids, const std::vector<float>* embd_row, int64_t n_rows,
                 int64_t first_pos, bool all_outputs, int32_t* rc_out) {
    llama_batch_ext* b = llama_batch_ext_init(ctx);
    if (b == nullptr) return false;
    bool ok = true;
    for (int64_t r = 0; r < n_rows && ok; ++r) {
        const int32_t idx = llama_batch_ext_add_token(b, 0, ids[r]);
        const llama_pos pos = (llama_pos) (first_pos + r);
        ok = idx >= 0 && llama_batch_ext_set_pos(b, idx, &pos);
        if (ok && embd_row != nullptr && r == 0) {
            const llama_embd e = {embd_row->data(), 1, embd_row->size()};
            ok = llama_batch_ext_set_embd_token(b, idx, e);
        }
        if (ok && (all_outputs || r == n_rows - 1)) ok = llama_batch_ext_set_output_logits(b, idx, true);
    }
    int32_t rc = -1;
    if (ok) rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    llama_batch_ext_free(b);
    if (rc_out != nullptr) *rc_out = rc;
    return ok;
}

void print_piece_hex(const llama_vocab* vocab, llama_token id, std::string* out) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    out->clear();
    for (int i = 0; i < (n > 0 ? n : 0); ++i) {
        char b[3];
        std::snprintf(b, sizeof(b), "%02x", (unsigned char) buf[i]);
        out->append(b);
    }
}

}  // namespace

bool qwen35_mtp_txn_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_TXN");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

int64_t qwen35_mtp_txn_k() {
    const char* v = std::getenv("STRATA_QWEN35_MTP_TXN_K");
    if (v == nullptr || v[0] == '\0') return 1;
    return std::strtoll(v, nullptr, 10);
}

void qwen35_mtp_txn_decide(llama_token draft_id, llama_token t0, llama_token t1, llama_token* emit,
                           llama_token* next_pending, int64_t* next_pos_delta, int64_t* rollback_rows) {
    if (t0 == draft_id) {
        *emit = draft_id;
        *next_pending = t1;
        *next_pos_delta = 2;
        *rollback_rows = 0;
    } else {
        *emit = t0;
        *next_pending = t0;
        *next_pos_delta = 1;
        *rollback_rows = 1;
    }
}

int qwen35_mtp_txn_run(llama_model* model, llama_context* tgt, const llama_vocab* vocab, int64_t first_pos,
                       llama_token first_token, int64_t max_new, std::vector<llama_token>* emitted_out,
                       Qwen35MtpTxnTiming* timing_out, std::string* err) {
    auto fail = [&](const std::string& what) {
        if (err != nullptr) *err = what;
        std::fprintf(stderr, "qwen35 mtp-txn: fail-closed: %s\n", what.c_str());
        std::fflush(stderr);
        return 1;
    };
    // Task 261/264: per-token and end-to-end wall observation inside the engine (additive event fields
    // only).  All values are relative to the engine entry (txn_begin); the caller composes them with its own
    // request anchor - they are never presented as the request E2E by themselves.
    const auto txn_begin = std::chrono::steady_clock::now();
    auto t_s = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - txn_begin).count();
    };
    Qwen35MtpTxnTiming* timing = timing_out;
    if (timing != nullptr) *timing = Qwen35MtpTxnTiming{};
    llama_token eog_token = -1;

    // Task 267: default-OFF stage cost accounting.  OFF => every phase cost stays 0 and no extra clock
    // read or synchronize is added, so the control path is the previous behavior (the two extra steady
    // reads per phase are inside the OFF guard).  ON adds host wall clocks only (plus the optional
    // registered sync), never a math/KV/accept/output change.
    static const bool cost_split_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_COST_SPLIT");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    static const bool cost_sync_on = cost_split_on && [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_COST_SYNC");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    // Task 268: exact STRATA_QWEN35_MTP_SERIAL_VERIFY=1 gate for the serial target verifier candidate.
    // OFF keeps the T2 target path (and with it every accept/reject/output decision) exactly as before; the
    // flag changes the verify decode structure only, never the greedy result, the emission or the risk data.
    static const bool serial_verify_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_SERIAL_VERIFY");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (timing != nullptr) {
        timing->cost_split_on = cost_split_on;
        timing->cost_sync_on = cost_sync_on;
    }
    // Host-wall accumulators (ms) + per-round reporting state.  INDEX: 0 init, 1 draft, 2 verify,
    // 3 hcopy, 4 replay, 5 rollback, 6 emit/host.  The clocks themselves (cost_phase / cost_write /
    // print_cost_line) are defined after the draft context exists - they capture it by reference.
    double ph[7] = {0, 0, 0, 0, 0, 0, 0};
    double prev[7] = {0, 0, 0, 0, 0, 0, 0};
    double prev_sync = 0.0;
    int64_t cost_sync_calls = 0;
    double cost_sync_wait_s = 0.0;
    int64_t round_idx = 0;
    double init_start_s = 0.0;

    if (emitted_out != nullptr) emitted_out->clear();
    if (model == nullptr || tgt == nullptr || vocab == nullptr) return fail("null model/ctx/vocab");
    if (max_new <= 0) return fail("max_new must be >= 1 visible token");
    if (qwen35_mtp_txn_k() != 1) return fail("K>1 not implemented in the greedy K=1 engine");
    if (llama_n_rs_seq(tgt) < 1) return fail("target context has n_rs_seq=0: rollback not provable");
    if (first_pos <= 0) return fail("no prompt token before the pending seed (first_pos <= 0)");
    llama_memory_t tmem = llama_get_memory(tgt);
    if (tmem == nullptr) return fail("target memory is null");

    const int64_t n_ctx = (int64_t) llama_n_ctx(tgt);
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t n_embd_out = (int64_t) llama_model_n_embd_out(model);
    if (first_pos >= n_ctx) return fail("first pending position >= n_ctx: refusing before any decode");

    bool ok = false;
    std::vector<float> pending_h = copy_nextn_row(tgt, (int32_t) (first_pos - 1), &ok);
    if (!ok) return fail("no finite tail nextn row for the last prompt token");
    std::fprintf(stderr, "qwen35 mtp-txn stage: pending_h row=%lld n_embd_out=%lld bytes=%zu\n",
                 (long long) (first_pos - 1), (long long) n_embd_out, pending_h.size() * sizeof(float));
    std::fflush(stderr);

    if (cost_split_on) init_start_s = t_s();
    auto dp = llama_context_default_params();
    dp.n_ctx = (uint32_t) n_ctx;
    dp.n_batch = 2;
    dp.n_ubatch = 2;
    dp.n_threads = 8;
    dp.n_threads_batch = 8;
    dp.op_offload = true;
    dp.n_seq_max = 1;
    dp.n_outputs_max = 1;
    dp.no_perf = true;
    dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dp.ctx_other = tgt;
    dp.n_rs_seq = 0;
    std::unique_ptr<llama_context, decltype(&llama_free)> draft(llama_init_from_model(model, dp), llama_free);
    if (draft == nullptr) return fail("draft MTP context init failed");
    llama_set_embeddings_nextn(draft.get(), true, true);

    // Task 267 clocks (defined here: cost_phase captures `draft`).  Every span is a control-flow
    // containment: a displaced readback or a diagnostic synchronize is INSIDE the phase span, not added
    // on top.  want_sync selects the diagnostic sync target: 0 none, +1 tgt, -1 draft.
    auto cost_phase = [&](int i, int8_t want_sync) {
        const double t = cost_split_on ? t_s() : 0.0;
        return [&, i, t, want_sync]() {
            if (!cost_split_on) return;
            if (cost_sync_on && want_sync != 0) {
                llama_context* c = want_sync > 0 ? tgt : draft.get();
                const double s0 = t_s();
                llama_synchronize(c);
                cost_sync_wait_s += t_s() - s0;
                ++cost_sync_calls;
            }
            ph[i] += (t_s() - t) * 1000.0;
        };
    };
    auto cost_write = [&]() {   // emit/host-write span
        const double t = cost_split_on ? t_s() : 0.0;
        return [&, t]() { if (cost_split_on) ph[6] += (t_s() - t) * 1000.0; };
    };
    auto print_cost_line = [&](const char* tag, int64_t round_no) {
        if (!cost_split_on) return;
        std::fprintf(stderr,
                     "qwen35 mtp-txn cost %s round=%lld d_init=%.3f d_draft=%.3f d_verify=%.3f d_hcopy=%.3f "
                     "d_replay=%.3f d_rollback=%.3f d_emit=%.3f c_init=%.3f c_draft=%.3f c_verify=%.3f "
                     "c_hcopy=%.3f c_replay=%.3f c_rollback=%.3f c_emit=%.3f sync_calls=%lld sync_wait_ms=%.3f\n",
                     tag, (long long) round_no, ph[0] - prev[0], ph[1] - prev[1], ph[2] - prev[2],
                     ph[3] - prev[3], ph[4] - prev[4], ph[5] - prev[5], ph[6] - prev[6], ph[0], ph[1],
                     ph[2], ph[3], ph[4], ph[5], ph[6], (long long) cost_sync_calls,
                     (cost_sync_wait_s - prev_sync) * 1000.0);
        std::fflush(stderr);
        for (int i = 0; i < 7; ++i) prev[i] = ph[i];
        prev_sync = cost_sync_wait_s;
    };
    {
        const double t = cost_split_on ? init_start_s : 0.0;
        if (cost_sync_on) {
            const double s0 = t_s();
            llama_synchronize(tgt);
            cost_sync_wait_s += t_s() - s0;
            ++cost_sync_calls;
        }
        if (cost_split_on) ph[0] += (t_s() - t) * 1000.0;
    }

    llama_token pending = first_token;
    const char* pending_role = "seed";   // seed | bonus | corrective
    int64_t pos = first_pos;
    int64_t rounds = 0;
    int64_t emitted = 0;
    int64_t eos_count = 0;
    const char* stop = "max_new";
    std::vector<llama_token> emits;

    auto emit_one = [&](llama_token id, int64_t at_pos, const char* role, int64_t round_no) {
        std::string piece;
        print_piece_hex(vocab, id, &piece);
        // ready = the token is decided, before its write.  Task 267 field correction: the old event
        // "flush_s" was evaluated as a printf ARGUMENT, i.e. BEFORE the write - it was never a flush
        // instant.  The raw line now carries only the honest pre-write t_s (plus round), and the real
        // post-fflush arrays travel in the typed emit_flush_s vector to the caller's request timing.
        // Old raw logs are untouched; nothing here re-measures or rewrites history.
        const double ready = t_s();
        if (timing != nullptr && timing->first_ready_s < 0) timing->first_ready_s = ready;
        const auto w_done = cost_write();
        std::printf("{\"event\":\"qwen35_mtp_txn_emit\",\"index\":%lld,\"round\":%lld,\"id\":%d,\"pos\":%lld,"
                    "\"role\":\"%s\",\"piece_hex\":\"%s\",\"t_s\":%.9g}\n",
                    (long long) emitted, (long long) round_no, (int) id, (long long) at_pos, role, piece.c_str(),
                    ready);
        std::fflush(stdout);
        const double flushed = t_s();
        w_done();
        if (timing != nullptr) {
            timing->emit_ready_s.push_back(ready);
            timing->emit_flush_s.push_back(flushed);
            if (timing->first_flush_s < 0) timing->first_flush_s = flushed;
        }
        emits.push_back(id);
        ++emitted;
    };

    while (true) {
        if (emitted >= max_new) { stop = "max_new"; break; }
        if (llama_vocab_is_eog(vocab, pending)) {
            stop = (std::strcmp(pending_role, "seed") == 0) ? "seed_eog" : "pending_eog";
            eog_token = pending;
            ++eos_count;
            break;
        }
        if (pos >= n_ctx) return fail("pending position >= n_ctx before decode");
        // The committed pending token is visible now: emit it exactly once (seed / accepted bonus /
        // corrective).  On reject the corrective is pending first and emitted here, not at reject time.
        emit_one(pending, pos, pending_role, round_idx);
        if (emitted >= max_new) { stop = "max_new"; break; }

        if (pos + 1 >= n_ctx) {
            // Context tail: a two-row verify would place the draft at n_ctx.  Legal final serial emit: one
            // one-row decode of the pending at pos (< n_ctx), emit its greedy successor, stop.
            int32_t rc = -1;
            const llama_token tail = pending;
            if (!decode_rows(tgt, &tail, nullptr, 1, pos, true, &rc) || rc != 0)
                return fail("ctx-tail serial decode failed");
            const float* lg = llama_get_logits_ith(tgt, -1);
            const llama_token t = greedy_argmax(lg, n_vocab, &ok);
            if (!ok) return fail("ctx-tail logits not finite");
            if (llama_vocab_is_eog(vocab, t)) { stop = "ctx_tail_eog"; eog_token = t; ++eos_count; break; }
            emit_one(t, pos + 1, "ctx_tail", round_idx);
            stop = "ctx_tail_serial";
            break;
        }

        int32_t rc = -1;
        const auto draft_phase_done = cost_phase(1, -1);
        if (!decode_rows(draft.get(), &pending, &pending_h, 1, pos, true, &rc) || rc != 0)
            return fail("draft round decode failed");
        const llama_token draft_id = greedy_argmax(llama_get_logits_ith(draft.get(), -1), n_vocab, &ok);
        if (!ok) return fail("draft logits not finite");
        draft_phase_done();

        llama_token t0 = -1;
        llama_token t1 = -1;   // serial: present only when T1#2 actually decoded the draft row (else -1)
        std::vector<float> h_row0;
        std::vector<float> h_row1;
        bool accepted = false;
        int64_t rollback = 0;   // rows actually removed from the target sequence this round

        if (!serial_verify_on) {
            // T2 control: one token-only target batch [pending, draft_id].  Every phase closes before the
            // next one opens (task 268 cost-window fix), so no span contains another span's work.
            const auto verify_phase_done = cost_phase(2, 1);
            const llama_token vids[2] = {pending, draft_id};
            if (!decode_rows(tgt, vids, nullptr, 2, pos, true, &rc) || rc != 0)
                return fail("target verify decode failed");
            t0 = greedy_argmax(llama_get_logits_ith(tgt, 0), n_vocab, &ok);
            if (!ok) return fail("verify row0 logits not finite");
            t1 = greedy_argmax(llama_get_logits_ith(tgt, 1), n_vocab, &ok);
            if (!ok) return fail("verify row1 logits not finite");
            verify_phase_done();
            // Deep-copy both verify rows BEFORE any further decode can touch the target's embedding buffers:
            // row0 = hidden after pending@pos (the h input for the accepted token replayed at pos+1), row1 =
            // hidden after draft@pos+1 (the next pending hidden after an accept).  The copy phase holds the
            // copies and their finiteness scans only - it is closed here, before any emit work.
            const auto hcopy_phase_done = cost_phase(3, 0);
            h_row0 = copy_nextn_row(tgt, 0, &ok);
            if (!ok) return fail("no finite nextn row0 for the accepted-prefix draft sync");
            h_row1 = copy_nextn_row(tgt, 1, &ok);
            if (!ok) return fail("no finite nextn row1 for the accepted bonus pending");
            hcopy_phase_done();
        } else {
            // Serial T1 verify (task 268, default OFF): T1#1 decodes the pending token alone; t0 and h0 are
            // read and copied immediately (before any later decode can refill the target buffers).  Only on
            // draft_id == t0 with a legal (non-EOG) draft token does T1#2 decode the draft token alone for
            // t1/h1.  On reject the target never decoded the speculative row, so nothing is removed.
            const auto verify1_phase_done = cost_phase(2, 1);
            const llama_token v0 = pending;
            if (!decode_rows(tgt, &v0, nullptr, 1, pos, true, &rc) || rc != 0)
                return fail("serial verify T1#1 (pending) decode failed");
            t0 = greedy_argmax(llama_get_logits_ith(tgt, -1), n_vocab, &ok);
            if (!ok) return fail("serial verify T1#1 logits not finite");
            verify1_phase_done();

            const auto hcopy0_phase_done = cost_phase(3, 0);
            h_row0 = copy_nextn_row(tgt, 0, &ok);
            if (!ok) return fail("serial verify: no finite h0 for the pending row");
            hcopy0_phase_done();

            accepted = (draft_id == t0);
            if (accepted && !llama_vocab_is_eog(vocab, draft_id)) {
                const auto verify2_phase_done = cost_phase(2, 1);
                const llama_token v1 = draft_id;
                if (!decode_rows(tgt, &v1, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                    return fail("serial verify T1#2 (draft) decode failed");
                t1 = greedy_argmax(llama_get_logits_ith(tgt, -1), n_vocab, &ok);
                if (!ok) return fail("serial verify T1#2 logits not finite");
                verify2_phase_done();

                const auto hcopy1_phase_done = cost_phase(3, 0);
                h_row1 = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("serial verify: no finite h1 for the draft row");
                hcopy1_phase_done();
            }
        }

        llama_token emit = -1, next_pending = -1;
        int64_t dpos = 0;
        if (!serial_verify_on) {
            qwen35_mtp_txn_decide(draft_id, t0, t1, &emit, &next_pending, &dpos, &rollback);
            accepted = (rollback == 0);
        } else if (accepted) {
            emit = draft_id;
            next_pending = t1;   // -1 only in the accepted-EOG stop below; never emitted
            dpos = 2;
        } else {
            emit = t0;
            next_pending = t0;
            dpos = 1;
        }
        const int64_t accept_report = accepted ? 1 : 0;

        std::string pe, pp;
        print_piece_hex(vocab, emit, &pe);
        print_piece_hex(vocab, next_pending, &pp);
        const auto round_write_done = cost_write();
        std::printf("{\"event\":\"qwen35_mtp_txn_round\",\"round\":%lld,\"pos\":%lld,\"seed\":%d,"
                    "\"draft\":%d,\"t0\":%d,\"t1\":%d,\"accept\":%lld,\"rollback\":%lld,\"emit\":%d,"
                    "\"piece_hex\":\"%s\",\"next_pending\":%d,\"next_piece_hex\":\"%s\","
                    "\"h_row0_bytes\":%zu,\"h_row1_bytes\":%zu,\"emitted_after\":%lld,\"t_s\":%.9g}\n",
                    (long long) rounds, (long long) pos, (int) pending, (int) draft_id, (int) t0, (int) t1,
                    (long long) accept_report, (long long) rollback, (int) emit, pe.c_str(),
                    (int) next_pending, pp.c_str(), h_row0.size() * sizeof(float), h_row1.size() * sizeof(float),
                    (long long) emitted, t_s());
        std::fflush(stdout);
        round_write_done();

        if (accepted) {
            // Accept: draft_id == t0 is the real greedy token at pos+1.  EOS stops without emitting it and
            // without any further replay; the target KV already holds the accepted prefix.
            if (llama_vocab_is_eog(vocab, draft_id)) { stop = "accepted_eog"; eog_token = draft_id; ++eos_count; break; }
            emit_one(draft_id, pos + 1, "draft", round_idx);
            if (emitted >= max_new) { stop = "max_new"; break; }
            // Draft KV sync of the accepted prefix: decode the accepted token at pos+1 WITH the deep-copied
            // target hidden after pending@pos (token+hidden, never token-only).  Its own logits are unused.
            const llama_token replay = draft_id;
            const auto replay_phase_done = cost_phase(4, -1);
            if (!decode_rows(draft.get(), &replay, &h_row0, 1, pos + 1, false, &rc) || rc != 0)
                return fail("draft KV sync of the accepted prefix failed");
            replay_phase_done();
            pending = t1;
            pending_h = std::move(h_row1);
            pending_role = "bonus";
            pos += dpos;   // 2
        } else {
            // Reject: the corrective t0 becomes the pending token and is emitted at the next round start,
            // exactly once.  T2 removed the speculative draft row it had decoded; the serial path never
            // decoded one (rollback = 0), so there is nothing to remove and seq_rm is skipped.
            if (rollback > 0) {
                const auto rollback_phase_done = cost_phase(5, 0);
                if (!llama_memory_seq_rm(tmem, 0, pos + 1, -1)) return fail("target rollback refused");
                rollback_phase_done();
            }
            pending = t0;
            pending_h = std::move(h_row0);
            pending_role = "corrective";
            pos += dpos;   // 1
        }
        print_cost_line("round", rounds);
        ++round_idx;
        ++rounds;
    }

    // The stop decision instant is final here; record it before the done event's own write.
    const double wall = t_s();
    if (timing != nullptr) {
        timing->ok = true;
        timing->engine_wall_s = wall;
        timing->rounds = rounds;
        timing->emitted = emitted;
        timing->eos_count = eos_count;
        timing->end_pos = pos;
        timing->eog_id = (long long) eog_token;
        timing->stop = stop;
    }
    // Task 268: the done event is written AFTER the stop decision, i.e. outside the finalized engine wall;
    // its write cost is therefore not attributed to any phase (the previous version counted it in emit
    // while the wall had already ended, which pushed the residual negative).
    std::printf("{\"event\":\"qwen35_mtp_txn_done\",\"rounds\":%lld,\"emitted\":%lld,\"eos\":%lld,"
                "\"stop\":\"%s\",\"end_pos\":%lld,\"max_new\":%lld,\"wall_s\":%.9g}\n",
                (long long) rounds, (long long) emitted, (long long) eos_count, stop, (long long) pos,
                (long long) max_new, wall);
    std::fflush(stdout);
    if (timing != nullptr && cost_split_on) {
        const double accounted_ms = ph[0] + ph[1] + ph[2] + ph[3] + ph[4] + ph[5] + ph[6];
        const double unattr_ms = wall * 1000.0 - accounted_ms;
        timing->ph_initdraftctx_ms = ph[0];
        timing->ph_draft_ms = ph[1];
        timing->ph_verify_ms = ph[2];
        timing->ph_hcopy_ms = ph[3];
        timing->ph_replay_ms = ph[4];
        timing->ph_rollback_ms = ph[5];
        timing->ph_emit_ms = ph[6];
        // Signed, no clamp (task 268): a negative residual honestly shows spans exceeding the wall - it is
        // never hidden by a floor.
        timing->ph_unattributed_ms = unattr_ms;
        timing->cost_sync_calls = cost_sync_calls;
        timing->cost_sync_wait_s = cost_sync_wait_s;
        print_cost_line("final", rounds);
    }
    if (emitted_out != nullptr) *emitted_out = std::move(emits);
    if (err != nullptr) err->clear();
    return 0;
}

}  // namespace strata::core
