// src/core/qwen35_mtp.cpp - task 235 (fixes over 232): the qwen35moe draft-layer reader/validator (default
// OFF).  Fix list from the 233/234 reviews: F32=0 sentinel (now a counted type list), attn_output name and
// [4096,2048] shape, ffn_gate_inp_shexp 1-D [2048], q/k norms [256], safe unknown-type rejection before the
// ggml type-name lookup, an INDEPENDENT positive fixture transcribed from the 229 header evidence, and
// exactly 16 self-test cases.  No draft generation/verify/rollback here.
#include "strata/core/qwen35_mtp.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {

// ggml type ids the draft block carries (ggml.h).
constexpr uint32_t kF32 = 0, kQ8_0 = 8, kQ2_K = 10, kQ4_K = 12, kQ6_K = 14, kBF16 = 30;
constexpr uint64_t kN = 2048;      // n_embd of the 35B-A3B main model
constexpr uint64_t kCtx = 4096;    // eh_proj / attn_output input width (2 * n_embd)
constexpr uint64_t kExp = 512;     // routed expert feed-forward width
constexpr uint64_t kNExp = 256;    // routed expert count
constexpr uint64_t kNHead = 256;   // attention head width for the q/k norms

using Actual = std::tuple<std::string, uint32_t, std::vector<uint64_t>>;

const std::vector<Qwen35DraftTensor>& schema_build() {
    static const std::vector<Qwen35DraftTensor> s = {
        {"nextn.enorm.weight", 1, {kN, 0, 0}, {kF32}, 1, true},
        {"nextn.hnorm.weight", 1, {kN, 0, 0}, {kF32}, 1, true},
        {"attn_norm.weight", 1, {kN, 0, 0}, {kF32}, 1, true},
        {"attn_q_norm.weight", 1, {kNHead, 0, 0}, {kF32}, 1, true},
        {"attn_k_norm.weight", 1, {kNHead, 0, 0}, {kF32}, 1, true},
        {"post_attention_norm.weight", 1, {kN, 0, 0}, {kF32}, 1, true},
        {"nextn.shared_head_norm.weight", 1, {kN, 0, 0}, {kF32}, 1, true},
        {"ffn_gate_inp.weight", 2, {kN, kNExp, 0}, {kBF16, kF32}, 2, true},
        {"ffn_gate_inp_shexp.weight", 1, {kN, 0, 0}, {kBF16, kF32}, 2, true},
        {"nextn.eh_proj.weight", 2, {kCtx, kN, 0}, {kQ8_0}, 1, true},
        {"attn_q.weight", 2, {kN, 0, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"attn_k.weight", 2, {kN, 0, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"attn_v.weight", 2, {kN, 0, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"attn_output.weight", 2, {kCtx, kN, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"ffn_gate_exps.weight", 3, {kN, kExp, kNExp}, {kQ2_K}, 1, true},
        {"ffn_up_exps.weight", 3, {kN, kExp, kNExp}, {kQ2_K}, 1, true},
        {"ffn_down_exps.weight", 3, {kExp, kN, kNExp}, {kQ4_K}, 1, true},
        {"ffn_gate_shexp.weight", 2, {kN, 0, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"ffn_up_shexp.weight", 2, {kN, 0, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
        {"ffn_down_shexp.weight", 2, {0, kN, 0}, {kF32, kBF16, kQ8_0, kQ6_K}, 4, true},
    };
    return s;
}

bool type_allowed(const Qwen35DraftTensor& d, uint32_t type) {
    for (int i = 0; i < d.n_types; ++i)
        if (d.types[i] == type) return true;
    return false;
}

const char* safe_type_name(uint32_t type) {
    if (type >= (uint32_t) GGML_TYPE_COUNT) return "unknown-type-id";
    return ggml_type_name((ggml_type) type);
}

std::string prefix_of(int64_t n_layers) { return "blk." + std::to_string(n_layers) + "."; }

// Independent positive fixture: the draft block's 20 tensors TRANSCRIBED from the task-229 header evidence
// (20/20 present; GU exps Q2_K, DOWN exps Q4_K, nextn.eh_proj Q8_0 [4096,2048]; norms F32; routers BF16;
// q/k norms [256]; ffn_gate_inp_shexp 1-D [2048]; attn_output [4096,2048]; the dense projections carry the
// family's dense quant type).  NOT generated from the schema above.
std::vector<Actual> actual_229() {
    const std::string p = prefix_of(40);
    return {
        {p + "nextn.enorm.weight", kF32, {kN}},
        {p + "nextn.hnorm.weight", kF32, {kN}},
        {p + "attn_norm.weight", kF32, {kN}},
        {p + "attn_q_norm.weight", kF32, {kNHead}},
        {p + "attn_k_norm.weight", kF32, {kNHead}},
        {p + "post_attention_norm.weight", kF32, {kN}},
        {p + "nextn.shared_head_norm.weight", kF32, {kN}},
        {p + "ffn_gate_inp.weight", kBF16, {kN, kNExp}},
        {p + "ffn_gate_inp_shexp.weight", kBF16, {kN}},
        {p + "nextn.eh_proj.weight", kQ8_0, {kCtx, kN}},
        {p + "attn_q.weight", kQ6_K, {kN, kN}},
        {p + "attn_k.weight", kQ6_K, {kN, kN}},
        {p + "attn_v.weight", kQ6_K, {kN, kN}},
        {p + "attn_output.weight", kQ6_K, {kCtx, kN}},
        {p + "ffn_gate_exps.weight", kQ2_K, {kN, kExp, kNExp}},
        {p + "ffn_up_exps.weight", kQ2_K, {kN, kExp, kNExp}},
        {p + "ffn_down_exps.weight", kQ4_K, {kExp, kN, kNExp}},
        {p + "ffn_gate_shexp.weight", kQ6_K, {kN, kExp}},
        {p + "ffn_up_shexp.weight", kQ6_K, {kN, kExp}},
        {p + "ffn_down_shexp.weight", kQ6_K, {kExp, kN}},
    };
}

}  // namespace

const std::vector<Qwen35DraftTensor>& qwen35_mtp_schema() { return schema_build(); }

Qwen35DraftReport qwen35_mtp_validate(const std::vector<Actual>& actual, int64_t n_layers) {
    Qwen35DraftReport rep;
    const std::string pfx = prefix_of(n_layers);
    for (const Qwen35DraftTensor& d : schema_build()) {
        const std::string name = pfx + d.suffix;
        const auto it = std::find_if(actual.begin(), actual.end(),
                                     [&](const auto& a) { return std::get<0>(a) == name; });
        if (it == actual.end()) {
            ++rep.missing;
            rep.checks.push_back({d.suffix, false, "missing tensor " + name});
            continue;
        }
        const uint32_t type = std::get<1>(*it);
        const std::vector<uint64_t>& shape = std::get<2>(*it);
        ++rep.checked;
        if ((int) shape.size() != d.ndim) {
            ++rep.mismatched;
            rep.checks.push_back({d.suffix, false,
                                  "ndim " + std::to_string(shape.size()) + " != " + std::to_string(d.ndim)});
            continue;
        }
        bool shape_ok = true;
        for (int i = 0; i < d.ndim; ++i)
            if (d.shape[i] != 0 && shape[(size_t) i] != d.shape[i]) shape_ok = false;
        if (!shape_ok) {
            ++rep.mismatched;
            rep.checks.push_back({d.suffix, false, "shape mismatch"});
            continue;
        }
        if (type >= (uint32_t) GGML_TYPE_COUNT) {
            ++rep.mismatched;
            rep.checks.push_back({d.suffix, false, "type id " + std::to_string(type) + " out of range"});
            continue;
        }
        if (!type_allowed(d, type)) {
            ++rep.mismatched;
            rep.checks.push_back({d.suffix, false, std::string("type ") + safe_type_name(type)});
            continue;
        }
        rep.checks.push_back({d.suffix, true, ""});
    }
    rep.ok = rep.missing == 0 && rep.mismatched == 0 && rep.checked == (int) schema_build().size();
    if (!rep.ok)
        rep.error = "draft layer rejected: " + std::to_string(rep.missing) + " missing, " +
                    std::to_string(rep.mismatched) + " mismatched, " + std::to_string(rep.checked) + "/" +
                    std::to_string(schema_build().size()) + " checked";
    return rep;
}

Qwen35DraftReport qwen35_mtp_probe(const std::string& gguf_path, int64_t n_layers) {
    Qwen35DraftReport rep;
    try {
        strata::GgufModel model({gguf_path});   // metadata mapping only; tensor payload is never read here
        const std::string pfx = prefix_of(n_layers);
        std::vector<Actual> actual;
        for (const Qwen35DraftTensor& d : schema_build()) {
            const std::string name = pfx + d.suffix;
            const TensorInfo* t = model.find(name, nullptr);
            if (t == nullptr) continue;
            actual.emplace_back(name, t->type, t->shape);
        }
        return qwen35_mtp_validate(actual, n_layers);
    } catch (const std::exception& e) {
        rep.ok = false;
        rep.error = std::string("draft probe failed: ") + e.what();
        return rep;
    }
}

int qwen35_mtp_selftest() {
    const std::string pfx = prefix_of(40);
    struct Case { const char* name; bool want; std::vector<Actual> v; };
    std::vector<Case> cases;

    cases.push_back({"valid_full_20_from_229", true, actual_229()});
    {
        auto v = actual_229();
        v.erase(v.begin());
        cases.push_back({"missing_one_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "nextn.eh_proj.weight") std::get<1>(a) = kQ6_K;
        cases.push_back({"eh_proj_wrong_type_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "nextn.eh_proj.weight") std::get<2>(a) = {2048, 2048};
        cases.push_back({"eh_proj_wrong_shape_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "ffn_gate_exps.weight") std::get<1>(a) = kQ4_K;
        cases.push_back({"gu_exps_wrong_type_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "ffn_down_exps.weight") std::get<1>(a) = kQ2_K;
        cases.push_back({"down_exps_wrong_type_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "nextn.enorm.weight") std::get<1>(a) = kBF16;
        cases.push_back({"norm_not_f32_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "ffn_gate_inp.weight") std::get<1>(a) = kQ6_K;
        cases.push_back({"router_wrong_type_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "attn_output.weight") std::get<2>(a) = {2048, 2048};
        cases.push_back({"attn_output_wrong_dim0_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "ffn_gate_exps.weight") std::get<2>(a) = {2048, 256};
        cases.push_back({"exps_ndim_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "ffn_down_exps.weight") std::get<2>(a) = {512, 2048, 128};
        cases.push_back({"exps_wrong_count_reject", false, v});
    }
    {
        auto v = actual_229();
        v.emplace_back(pfx + "attn_extra.weight", kQ8_0, std::vector<uint64_t>{2048, 2048});
        cases.push_back({"extra_tensors_ignored_pass", true, v});
    }
    cases.push_back({"empty_list_reject", false, {}});
    {
        std::vector<Actual> v = actual_229();
        for (auto& a : v) std::get<0>(a) = prefix_of(41) + std::get<0>(a).substr(pfx.size());
        cases.push_back({"wrong_layer_prefix_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "nextn.hnorm.weight") std::get<2>(a) = {1024};
        cases.push_back({"norm_wrong_width_reject", false, v});
    }
    {
        auto v = actual_229();
        for (auto& a : v) if (std::get<0>(a) == pfx + "attn_q.weight") std::get<1>(a) = 44;
        cases.push_back({"unknown_type_id_reject", false, v});
    }

    int failed = 0;
    for (const Case& c : cases) {
        const Qwen35DraftReport r = qwen35_mtp_validate(c.v, 40);
        const bool ok = r.ok == c.want;
        if (!ok) ++failed;
        std::printf("{\"event\":\"qwen35_mtp_selftest\",\"case\":\"%s\",\"want\":%s,\"got\":%s,\"ok\":%s}\n",
                    c.name, c.want ? "true" : "false", r.ok ? "true" : "false", ok ? "true" : "false");
    }
    std::printf("{\"event\":\"qwen35_mtp_selftest\",\"cases\":%d,\"failed\":%d}\n", (int) cases.size(), failed);
    std::fflush(stdout);
    return failed == 0 ? 0 : 1;
}

bool qwen35_mtp_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

bool qwen35_mtp_selftest_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_SELFTEST");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

bool qwen35_mtp_draft_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_DRAFT");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

}  // namespace strata::core
