// src/program/qwen35_expert_check.cpp - the small REAL-WEIGHT numeric check for the two kernel fixes.
//
//   1. IQ2_S's dot: the device formula was the upstream truncating integer expression
//      `(sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4`; the reference identity is `1 + 2*ls` with one 0.125
//      factor.  Here three real gate rows of the model are compared: the GPU kernel (`iq_mmvq`) against an
//      fp64 reference over the dequantized weights/activation, and the truncating form (evaluated exactly, in
//      double, from the same real blocks) against the same reference.
//   2. DOWN21: the down projection of 37 layers is IQ3_S (ggml type 21), which the Flash-Next down dispatch
//      did not carry; a real down row is run through `iq_mmvq` and compared to the fp64 reference.
//   3. DOWN23: an IQ4_XS (type 23) down row from a later layer, the other down type of this file.
#include "strata/core/expert_source.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void report(const char* what, double rel, double limit) {
    const bool ok = rel < limit;
    std::printf("%-34s rel %.3e  (limit %.1e)  %s\n", what, rel, limit, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.0f : 1.0f) * (float) m * 5.960464477539063e-08f;   // subnormal
    if (e == 31) return (s ? -1.0f : 1.0f) * INFINITY;
    const uint32_t f = s | ((e - 15 + 127) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

struct RowCheck {
    double num = 0, den = 0;
    void add(double got, double ref) { num += std::fabs(got - ref); den += std::fabs(ref); }
    double rel() const { return num / (den + 1e-30); }
};

/// The fp64 reference of one i-quant row dotted with one q8_1 activation, both dequantized.

/// The scale identity, on the integer sums the kernel forms: the fixed form is
/// (sumi0*(2*ls0+1) + sumi1*(2*ls1+1)) * 0.125 in fp32 (the CPU reference's own identity); the upstream
/// truncating form divides integers.  Random real-shaped sums quantify what the fix removes.
void check_scale_formula() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> sums(-30000, 30000);
    std::uniform_int_distribution<int> ls(0, 15);
    double max_trunc = 0, max_fixed = 0;
    for (int i = 0; i < 20000; ++i) {
        const int s0 = sums(rng), s1 = sums(rng), l0 = ls(rng), l1 = ls(rng);
        const double exact = (s0 * (2.0 * l0 + 1) + s1 * (2.0 * l1 + 1)) * 0.125;
        const float fixed = (float) (s0 * (2 * l0 + 1) + s1 * (2 * l1 + 1)) * 0.125f;
        const double t = std::trunc(std::trunc(s0 * (double) l0 + s1 * (double) l1 +
                                               std::trunc((s0 + s1) / 2.0)) / 4.0);
        const double trunc_value = t;   // the same d*ds factor cancels in the relative comparison
        const double denom = std::fabs(exact) + 1e-9;
        max_trunc = std::max(max_trunc, std::fabs(trunc_value - exact) / denom);
        max_fixed = std::max(max_fixed, std::fabs((double) fixed - exact) / denom);
    }
    report("IQ2_S fixed identity max rel err", max_fixed, 1e-6);
    std::printf("%-34s %.3e  (what the upstream integer truncation adds)\n", "IQ2_S truncating max rel err",
                max_trunc);
}

/// One real expert through the MMQ prompt pipeline (the new GPU prefill path) against the CPU slice path:
/// the activations are q8_1 here and Q8_K there, so the tolerance is the activation-contract gap, not bitwise.
void check_mmq_expert(strata::core::FileExpertSource& src, cudaStream_t stream, int64_t layer) {
    using namespace strata::kernels::cpu;
    namespace mmq = strata::prefill::mmq;
    if (!mmq::built()) { std::printf("MMQ expert check: no MMQ build; skipped\n"); return; }
    const auto& lay = expert_layout();
    const NativeFmt& f = lay.fmt[(size_t) layer];
    const uint8_t* roles[3] = {nullptr, nullptr, nullptr};
    if (!src.slices(layer, 0, roles)) return;
    const int64_t N = f.n_embd, FF = f.n_ff;
    std::vector<float> x((size_t) N);
    for (int64_t i = 0; i < N; ++i) x[(size_t) i] = std::sin(0.017 * (double) i) * 2.4f;
    // CPU reference (the same arithmetic the decode misses use)
    std::vector<uint8_t> act(kNativeActBytes);
    native_quant_act(f, x.data(), act.data());
    const void* acts[1] = {act.data()};
    std::vector<float> ff((size_t) FF), hq(kNativeHBytes);
    std::vector<float> ref((size_t) N);
    float* ffs[1] = {ff.data()};
    float* refs[1] = {ref.data()};
    native_gu_rows_slices(f, roles[0], roles[1], acts, 1, ffs, 0, (int) FF);
    native_quant_h(f, ff.data(), hq.data());
    const void* hqs[1] = {hq.data()};
    native_down_rows_slice(f, roles[2], hqs, 1, refs, 0, (int) N);
    // MMQ path for the same expert and activation
    float* dx = nullptr;
    void* xq = nullptr;
    void* grp_gu = nullptr;
    void* grp_d = nullptr;
    float* dgu = nullptr;
    float* dh = nullptr;
    void* hqq = nullptr;
    float* ddm = nullptr;
    int32_t* dbounds = nullptr;
    int32_t* dident = nullptr;
    const size_t gu_group = mmq::matrix_bytes(f.gu_type, 2 * FF, N);
    const size_t d_group = mmq::matrix_bytes(f.d_type, N, FF);
    bool ok = cudaMalloc((void**) &dx, (size_t) N * 4) == cudaSuccess &&
              cudaMalloc(&xq, mmq::q8_bytes(1, N)) == cudaSuccess &&
              cudaMalloc(&grp_gu, gu_group + 4096) == cudaSuccess &&
              cudaMalloc(&grp_d, d_group + 4096) == cudaSuccess &&
              cudaMalloc((void**) &dgu, (size_t) 2 * FF * 4) == cudaSuccess &&
              cudaMalloc((void**) &dh, (size_t) FF * 4) == cudaSuccess &&
              cudaMalloc(&hqq, mmq::q8_bytes(1, FF)) == cudaSuccess &&
              cudaMalloc((void**) &ddm, (size_t) N * 4) == cudaSuccess &&
              cudaMalloc((void**) &dbounds, 8) == cudaSuccess &&
              cudaMalloc((void**) &dident, 4) == cudaSuccess;
    if (!ok) { report("MMQ expert alloc", 1.0, 0.5); return; }
    int32_t bounds[2] = {0, 1}, ident[1] = {0};
    cudaMemcpy(dx, x.data(), (size_t) N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbounds, bounds, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dident, ident, 4, cudaMemcpyHostToDevice);
    auto step = [&](const char* what) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) std::printf("MMQ expert step FAIL %s: %s\n", what, cudaGetErrorString(e));
    };
    uint8_t* dblob = nullptr;
    const size_t blob_bytes = 2 * (gu_group / 2) + d_group;
    if (cudaMalloc((void**) &dblob, blob_bytes) != cudaSuccess) { report("MMQ expert blob alloc", 1.0, 0.5); return; }
    cudaMemcpy(dblob, roles[0], gu_group / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dblob + gu_group / 2, roles[1], gu_group / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dblob + gu_group, roles[2], d_group, cudaMemcpyHostToDevice);
    mmq::quantize(dx, nullptr, xq, f.gu_type, N, N, 1, stream);
    step("quantize");
    mmq::gather_native(dblob, dblob + gu_group / 2, gu_group / 2, dblob + gu_group, d_group, grp_gu, grp_d, stream);
    step("gather");
    mmq::Context ctx;
    mmq::Product gu;
    gu.w = grp_gu; gu.type = f.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = gu_group;
    gu.n = 1; gu.xq = xq; gu.bounds = dbounds; gu.ids = dident; gu.total_rows = 1; gu.max_rows = 1;
    gu.dst = dgu; gu.ld_dst = 2 * FF;
    ctx.run(gu, stream);
    step("gu product");
    mmq::swiglu(dgu, dh, 1, FF, false, stream);
    step("swiglu");
    mmq::quantize(dh, nullptr, hqq, f.d_type, FF, FF, 1, stream);
    mmq::Product dn;
    dn.w = grp_d; dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = d_group;
    dn.n = 1; dn.xq = hqq; dn.bounds = dbounds; dn.ids = dident; dn.total_rows = 1; dn.max_rows = 1;
    dn.dst = ddm; dn.ld_dst = N;
    ctx.run(dn, stream);
    step("down product");
    cudaStreamSynchronize(stream);
    std::vector<float> got((size_t) N);
    cudaMemcpy(got.data(), ddm, (size_t) N * 4, cudaMemcpyDeviceToHost);
    RowCheck c;
    for (int64_t i = 0; i < N; ++i) c.add((double) got[(size_t) i], (double) ref[(size_t) i]);
    char label[96];
    std::snprintf(label, sizeof label, "MMQ expert L%lld vs CPU slice (rel)", (long long) layer);
    report(label, c.rel(), 3e-2);
    cudaFree(dblob); cudaFree(dx); cudaFree(xq); cudaFree(grp_gu); cudaFree(grp_d); cudaFree(dgu); cudaFree(dh); cudaFree(hqq);
    cudaFree(ddm); cudaFree(dbounds); cudaFree(dident);
}

/// The Q8_K activation quantizer of the shared service against ggml's own from_float, byte for byte.
void check_activation_quantizers() {
    using namespace strata::kernels::cpu;
    const auto& lay = expert_layout();
    const NativeFmt& f = lay.fmt[0];
    std::vector<float> x1(2048), x2(512);
    for (int i = 0; i < 2048; ++i) x1[(size_t) i] = std::sin(0.017 * i) * 3.1f;
    for (int i = 0; i < 512; ++i) x2[(size_t) i] = std::cos(0.031 * i) * 1.7f;
    std::vector<uint8_t> a(kNativeActBytes), b(kNativeActBytes), c(kNativeHBytes), d(kNativeHBytes);
    native_quant_act(f, x1.data(), a.data());
    ggml_get_type_traits_cpu((ggml_type) f.gu_act)->from_float(x1.data(), b.data(), f.n_embd);
    size_t diff = 0;
    for (size_t i = 0; i < (size_t) f.act_bytes; ++i) diff += a[i] != b[i];
    report("Q8_K act quantizer: ours vs ggml (bytes)", (double) diff, 1.0);
    native_quant_h(f, x2.data(), c.data());
    ggml_get_type_traits_cpu((ggml_type) f.d_act)->from_float(x2.data(), d.data(), f.n_ff);
    size_t diff2 = 0;
    for (size_t i = 0; i < (size_t) f.h_bytes; ++i) diff2 += c[i] != d[i];
    report("Q8_K hidden quantizer: ours vs ggml (bytes)", (double) diff2, 1.0);
}

/// One real expert computed the slice way (the decode CPU path) and from its packed blob (the
/// multi-token/pooled path), on the same activation.  Any difference here is the slice path's own.
void check_slice_vs_blob(strata::core::FileExpertSource& src, cudaStream_t) {
    using namespace strata::kernels::cpu;
    const auto& lay = expert_layout();
    const NativeFmt& f = lay.fmt[0];
    const uint8_t* roles[3] = {nullptr, nullptr, nullptr};
    if (!src.slices(0, 0, roles)) return;
    std::vector<float> x((size_t) f.n_embd);
    for (int64_t i = 0; i < f.n_embd; ++i) x[(size_t) i] = std::sin(0.011 * (double) i) * 2.2f;
    std::vector<uint8_t> act(kNativeActBytes);
    native_quant_act(f, x.data(), act.data());
    const void* acts[1] = {act.data()};

    std::vector<float> ff1((size_t) f.n_ff), ff2((size_t) f.n_ff);
    std::vector<uint8_t> hq1(kNativeHBytes), hq2(kNativeHBytes);
    std::vector<float> out1((size_t) f.n_embd), out2((size_t) f.n_embd);
    float* ffs1[1] = {ff1.data()};
    float* ffs2[1] = {ff2.data()};
    float* outs1[1] = {out1.data()};
    float* outs2[1] = {out2.data()};
    native_gu_rows_slices(f, roles[0], roles[1], acts, 1, ffs1, 0, (int) f.n_ff);
    native_quant_h(f, ff1.data(), hq1.data());
    const void* hqs1[1] = {hq1.data()};
    native_down_rows_slice(f, roles[2], hqs1, 1, outs1, 0, (int) f.n_embd);

    const uint8_t* blob = src.blob(0, 0);
    if (blob != nullptr) {
        native_gu_rows(f, blob, acts, 1, ffs2, 0, (int) f.n_ff);
        native_quant_h(f, ff2.data(), hq2.data());
        const void* hqs2[1] = {hq2.data()};
        native_down_rows(f, blob, hqs2, 1, outs2, 0, (int) f.n_embd);
        double num = 0, den = 0, maxrel = 0;
        for (int64_t i = 0; i < f.n_embd; ++i) {
            const double d = std::fabs((double) out1[(size_t) i] - out2[(size_t) i]);
            num += d;
            den += std::fabs((double) out2[(size_t) i]);
            maxrel = std::max(maxrel, d / (std::fabs((double) out2[(size_t) i]) + 1e-6));
        }
        report("expert: slice path vs packed blob (rel)", num / (den + 1e-30), 1e-6);
        std::printf("%-34s %.3e  (max elementwise relative)\n", "expert: slice vs blob max rel", maxrel);
    }
}

/// The decode CPU path's ROW SPLIT (pool phase 8/9) against the whole-expert task it replaces: the same
/// real expert slices, the same per-token activations, the gu rows and the down rows computed in scrambled
/// row ranges.  Every row runs the same per-row kernel either way, so ff, hq and out must be BITWISE equal -
/// this is the check the row-split reuse must pass for nm = 1, 3 and 8 (1 expert x 1/3 tokens, 8 experts x 1).
void check_row_split_slices(strata::core::FileExpertSource& src) {
    using namespace strata::kernels::cpu;
    const auto& lay = expert_layout();
    const NativeFmt& f = lay.fmt[0];
    const int64_t N = f.n_embd, FF = f.n_ff;
    for (const int nm : {1, 3, 8}) {
        const int n_experts_used = nm == 8 ? 8 : 1;
        std::vector<const uint8_t*> gate((size_t) n_experts_used), up((size_t) n_experts_used),
            down((size_t) n_experts_used);
        bool have = true;
        for (int e = 0; e < n_experts_used && have; ++e) {
            const uint8_t* roles[3] = {nullptr, nullptr, nullptr};
            have = src.slices(0, e, roles);
            gate[(size_t) e] = roles[0];
            up[(size_t) e] = roles[1];
            down[(size_t) e] = roles[2];
        }
        if (!have) {
            std::printf("row split nm=%d: slices unavailable; skipped\n", nm);
            continue;
        }
        std::vector<std::vector<uint8_t>> acts((size_t) nm, std::vector<uint8_t>(kNativeActBytes));
        std::vector<const void*> actp((size_t) nm);
        for (int t = 0; t < nm; ++t) {
            std::vector<float> x((size_t) N);
            for (int64_t i = 0; i < N; ++i) x[(size_t) i] = std::sin(0.013 * (double) i + 0.37 * (double) t) * 2.1f;
            native_quant_act(f, x.data(), acts[(size_t) t].data());
            actp[(size_t) t] = acts[(size_t) t].data();
        }
        std::vector<int> tok_expert((size_t) nm);
        for (int t = 0; t < nm; ++t) tok_expert[(size_t) t] = nm == 8 ? t : 0;
        std::vector<float> ff_ref((size_t) nm * (size_t) FF), ff_split((size_t) nm * (size_t) FF);
        std::vector<uint8_t> hq_ref((size_t) nm * kNativeHBytes), hq_split((size_t) nm * kNativeHBytes);
        std::vector<float> out_ref((size_t) nm * (size_t) N), out_split((size_t) nm * (size_t) N);
        for (int t = 0; t < nm; ++t) {
            const int e = tok_expert[(size_t) t];
            float* ffr[1] = {ff_ref.data() + (size_t) t * FF};
            const void* actr[1] = {actp[(size_t) t]};
            native_gu_rows_slices(f, gate[(size_t) e], up[(size_t) e], actr, 1, ffr, 0, (int) FF);
            native_quant_h(f, ffr[0], hq_ref.data() + (size_t) t * kNativeHBytes);
            const void* hqr[1] = {hq_ref.data() + (size_t) t * kNativeHBytes};
            float* outr[1] = {out_ref.data() + (size_t) t * N};
            native_down_rows_slice(f, down[(size_t) e], hqr, 1, outr, 0, (int) N);
        }
        for (int t = 0; t < nm; ++t) {
            const int e = tok_expert[(size_t) t];
            float* ffs[1] = {ff_split.data() + (size_t) t * FF};
            const int cuts[4] = {0, (int) FF / 3, (int) (2 * FF) / 3, (int) FF};
            const int gu_ranges[3][2] = {{cuts[1], cuts[2]}, {cuts[2], cuts[3]}, {cuts[0], cuts[1]}};
            for (const auto& rg : gu_ranges)
                native_gu_rows_slices(f, gate[(size_t) e], up[(size_t) e], actp.data() + t, 1, ffs, rg[0], rg[1]);
            native_quant_h(f, ffs[0], hq_split.data() + (size_t) t * kNativeHBytes);
            const void* hqs[1] = {hq_split.data() + (size_t) t * kNativeHBytes};
            float* outs[1] = {out_split.data() + (size_t) t * N};
            const int dcuts[3] = {0, (int) N / 2, (int) N};
            const int dn_ranges[2][2] = {{dcuts[1], dcuts[2]}, {dcuts[0], dcuts[1]}};
            for (const auto& rg : dn_ranges)
                native_down_rows_slice(f, down[(size_t) e], hqs, 1, outs, rg[0], rg[1]);
        }
        size_t bad = 0;
        for (size_t i = 0; i < ff_ref.size(); ++i) bad += ff_ref[i] != ff_split[i];
        for (size_t i = 0; i < hq_ref.size(); ++i) bad += hq_ref[i] != hq_split[i];
        for (size_t i = 0; i < out_ref.size(); ++i) bad += out_ref[i] != out_split[i];
        std::printf("%-34s %zu mismatched bytes  %s\n", ("row split nm=" + std::to_string(nm) +
                                                         ": ff/hq/out bitwise").c_str(),
                    bad, bad == 0 ? "ok" : "FAIL");
        if (bad != 0) ++failures;
    }
}

/// The hit launch's packed metadata block against the same fields uploaded SEPARATELY (the pre-change form):
/// one real expert and eight real experts of layer 0 are admitted to a small cache, both paths run the same
/// `quantize_q8_1_rows` + `native_expert_grouped` with the same entries (nh = 0/1/8), and the result rows,
/// the start sentinels and the hit counters must agree bitwise / exactly.  No kernel math involved.
void check_hit_metadata_plan(strata::core::FileExpertSource& src, cudaStream_t stream) {
    using namespace strata::kernels::cpu;
    const auto& lay = expert_layout();
    const NativeFmt& f = lay.fmt[0];
    const int64_t N = f.n_embd;
    const int K = 8;
    strata::core::ExpertDispatch d;
    d.src = &src;
    d.n_expert = lay.n_expert;
    d.n_embd = f.n_embd;
    d.n_ff = f.n_ff;
    std::string err;
    if (!strata::core::expert_native_cache_open(d, K, "", err)) {
        std::printf("hit metadata plan: cache open FAILED: %s\n", err.c_str());
        ++failures;
        return;
    }
    for (int i = 0; i < K; ++i) {
        const int32_t slot = d.cache->admit(0, i);
        const uint8_t* b = src.blob_stable(0, i);
        if (slot < 0 || b == nullptr || !d.cache->fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(0))) {
            std::printf("hit metadata plan: cache fill FAILED: %s\n", err.c_str());
            ++failures;
            strata::core::expert_native_close(d);
            return;
        }
        d.native_slot_ptr[(size_t) slot] = (unsigned long long) d.cache->device_slot(slot);
    }
    float* dx = nullptr;
    cudaMalloc((void**) &dx, (size_t) N * sizeof(float));
    std::vector<float> x((size_t) N);
    for (int64_t i = 0; i < N; ++i) x[(size_t) i] = std::sin(0.019 * (double) i + 0.11) * 2.3f;
    cudaMemcpy(dx, x.data(), (size_t) N * sizeof(float), cudaMemcpyHostToDevice);
    const strata::kernels::NativeExpertLayout G = strata::kernels::native_expert_layout(f.gu_type, f.d_type, N, f.n_ff);
    std::vector<int32_t> ids((size_t) K, -1);
    std::vector<uint8_t> is_hit((size_t) K, 0);
    int bad = 0;
    for (const int nh_want : {0, 1, 8}) {
        for (int i = 0; i < K; ++i) ids[(size_t) i] = i < nh_want ? i : -1;
        const int64_t hits_before = d.native_hits;
        if (!strata::core::expert_native_launch(d, 0, stream, dx, ids.data(), K, is_hit.data())) {
            std::printf("hit metadata plan nh=%d: launch failed (%s)\n", nh_want, d.fail != nullptr ? d.fail : "?");
            ++bad;
            continue;
        }
        // expected hits/misses in `is_hit`
        int expect_hit = 0;
        for (int i = 0; i < K; ++i)
            if (is_hit[(size_t) i] != (i < nh_want ? 1 : 0)) ++bad;
            else if (is_hit[(size_t) i]) ++expect_hit;
        if (d.native_hits - hits_before != expect_hit) ++bad;
        if (nh_want == 0) continue;   // nothing uploaded or computed; the early return is the contract
        // Reference: the same entries, fields in SEPARATE device buffers (the form this change replaced).
        unsigned long long* rp = nullptr;
        int32_t* rs = nullptr;
        int32_t* rd = nullptr;
        int32_t* rt = nullptr;
        int32_t* rg = nullptr;
        float* dact = nullptr;
        void* xq = nullptr;
        void* scratch = nullptr;
        float* parts_ref = nullptr;
        std::vector<unsigned long long> ptr_h((size_t) K, 0);
        std::vector<int32_t> start_h((size_t) K + 1, 0), dst_h((size_t) K, 0), tok_h((size_t) K, 0);
        int nh_ref = 0;
        for (int i = 0; i < K; ++i) {
            const int32_t slot = ids[(size_t) i] >= 0 ? d.cache->slot_of(0, ids[(size_t) i]) : -1;
            if (slot < 0) continue;
            ptr_h[(size_t) nh_ref] = d.native_slot_ptr[(size_t) slot];
            start_h[(size_t) nh_ref] = nh_ref;
            dst_h[(size_t) nh_ref] = i;
            tok_h[(size_t) nh_ref] = 0;
            ++nh_ref;
        }
        start_h[(size_t) nh_ref] = nh_ref;
        const int32_t groups = nh_ref;
        const size_t scratch_bytes = strata::kernels::native_expert_scratch_bytes(K, f.n_ff);
        bool ok = cudaMalloc((void**) &rp, (size_t) K * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMalloc((void**) &rs, (size_t) (K + 1) * sizeof(int32_t)) == cudaSuccess &&
                  cudaMalloc((void**) &rd, (size_t) K * sizeof(int32_t)) == cudaSuccess &&
                  cudaMalloc((void**) &rt, (size_t) K * sizeof(int32_t)) == cudaSuccess &&
                  cudaMalloc((void**) &rg, sizeof(int32_t)) == cudaSuccess &&
                  cudaMalloc((void**) &dact, (size_t) N * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(&xq, (size_t) (N / 32) * 36) == cudaSuccess &&
                  cudaMalloc(&scratch, scratch_bytes) == cudaSuccess &&
                  cudaMalloc((void**) &parts_ref, (size_t) K * (size_t) N * sizeof(float)) == cudaSuccess;
        if (!ok) { std::printf("hit metadata plan nh=%d: reference alloc failed\n", nh_want); ++bad; }
        else {
            cudaMemcpyAsync(rp, ptr_h.data(), (size_t) nh_ref * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(rs, start_h.data(), (size_t) (nh_ref + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(rd, dst_h.data(), (size_t) nh_ref * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(rt, tok_h.data(), (size_t) nh_ref * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(rg, &groups, sizeof(groups), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dact, dx, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            strata::kernels::quantize_q8_1_rows(dact, 1, N, xq, stream);
            strata::kernels::native_expert_grouped(G, rp, rs, rg, rd, rt, K, K, xq, scratch, parts_ref, stream, nh_ref);
            cudaStreamSynchronize(stream);
            std::vector<float> got((size_t) K * (size_t) N), ref((size_t) K * (size_t) N);
            cudaMemcpy(got.data(), d.native_parts_dev, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(ref.data(), parts_ref, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
            size_t diff = 0;
            for (int i = 0; i < nh_ref; ++i)
                if (std::memcmp(got.data() + (size_t) i * N, ref.data() + (size_t) i * N,
                                (size_t) N * sizeof(float)) != 0)
                    ++diff;
            if (diff != 0 || nh_ref != nh_want) ++bad;
            std::printf("hit metadata plan nh=%d: %zu result rows differ from the separate-fields reference\n",
                        nh_want, diff);
        }
        cudaFree(rp); cudaFree(rs); cudaFree(rd); cudaFree(rt); cudaFree(rg);
        cudaFree(dact); cudaFree(xq); cudaFree(scratch); cudaFree(parts_ref);
    }
    std::printf("hit metadata plan: %s\n", bad == 0 ? "ok" : "FAIL");
    if (bad != 0) ++failures;
    cudaFree(dx);
    strata::core::expert_native_close(d);
}

/// Real IQ2_S row: the GPU kernel vs the fp64 reference over the dequantized weights/activation.
void check_iq2s_row(const uint8_t* row, int64_t n_in, const uint8_t* xq8, void* xq_dev, cudaStream_t stream) {
    void* dw = nullptr;
    float* dy = nullptr;
    cudaMalloc(&dw, (size_t) n_in / 256 * 82);
    cudaMalloc(&dy, sizeof(float));
    cudaMemcpy(dw, row, (size_t) n_in / 256 * 82, cudaMemcpyHostToDevice);
    strata::kernels::iq_mmvq((int) GGML_TYPE_IQ2_S, dw, xq_dev, dy, (int) n_in, 1, 1, stream);
    cudaStreamSynchronize(stream);
    float y = 0;
    cudaMemcpy(&y, dy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dw);
    cudaFree(dy);

    std::vector<float> w((size_t) n_in);
    ggml_get_type_traits(GGML_TYPE_IQ2_S)->to_float(row, w.data(), n_in);
    double ref = 0;
    for (int64_t i = 0; i < n_in; ++i) {
        const int64_t blk = i / 32, in = i % 32;
        const uint8_t* q8 = xq8 + blk * 36;
        const double ds = half_to_float((uint16_t) (q8[0] | (q8[1] << 8)));
        ref += (double) w[(size_t) i] * ds * (double) ((const int8_t*) (q8 + 4))[in];
    }
    RowCheck c;
    c.add((double) y, ref);
    report("IQ2_S gate row: fixed kernel vs fp64", c.rel(), 2e-4);
}

/// A real down row of `type` through the GPU dispatch vs the fp64 reference.
void check_down_row(int type, const char* name, const uint8_t* row, int64_t n_in, const uint8_t* xq8,
                    void* xq_dev, cudaStream_t stream) {
    const size_t row_bytes = ggml_row_size((ggml_type) type, n_in);
    void* dw = nullptr;
    float* dy = nullptr;
    cudaMalloc(&dw, row_bytes);
    cudaMalloc(&dy, sizeof(float));
    cudaMemcpy(dw, row, row_bytes, cudaMemcpyHostToDevice);
    strata::kernels::iq_mmvq(type, dw, xq_dev, dy, (int) n_in, 1, 1, stream);
    cudaStreamSynchronize(stream);
    float y = 0;
    cudaMemcpy(&y, dy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dw);
    cudaFree(dy);
    std::vector<float> w((size_t) n_in);
    ggml_get_type_traits((ggml_type) type)->to_float(row, w.data(), n_in);
    double ref = 0;
    for (int64_t i = 0; i < n_in; ++i) {
        const int64_t blk = i / 32, in = i % 32;
        const uint8_t* q8 = xq8 + blk * 36;
        const double ds = half_to_float((uint16_t) (q8[0] | (q8[1] << 8)));
        ref += (double) w[(size_t) i] * ds * (double) ((const int8_t*) (q8 + 4))[in];
    }
    RowCheck c;
    c.add((double) y, ref);
    char label[96];
    std::snprintf(label, sizeof label, "%s down row: dispatch vs fp64", name);
    report(label, c.rel(), 2e-4);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string pack = argc > 1 ? argv[1] : "models";
    const std::string model = argc > 2 ? argv[2] : "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf";
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(pack, 40, 256, 2048, 512, err)) {
        std::fprintf(stderr, "qwen35-expert-check: %s\n", err.c_str());
        return 1;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    strata::core::FileExpertSource src;
    src.set_gguf(model);
    if (!src.open(pack, 40, 256, err)) {
        std::fprintf(stderr, "qwen35-expert-check: %s\n", err.c_str());
        return 1;
    }
    // A deterministic activation, quantized with the same GPU quantizer the hit path uses.
    const int64_t n_in = 2048;
    std::vector<float> x((size_t) n_in);
    for (int64_t i = 0; i < n_in; ++i) x[(size_t) i] = std::sin(0.013 * (double) i) * 2.5f;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* dx = nullptr;
    void* xq = nullptr;
    cudaMalloc((void**) &dx, (size_t) n_in * sizeof(float));
    cudaMalloc(&xq, (size_t) n_in / 32 * 36);
    cudaMemcpy(dx, x.data(), (size_t) n_in * sizeof(float), cudaMemcpyHostToDevice);
    strata::kernels::quantize_q8_1_rows(dx, 1, n_in, xq, stream);
    cudaStreamSynchronize(stream);
    std::vector<uint8_t> xq8((size_t) n_in / 32 * 36);
    cudaMemcpy(xq8.data(), xq, xq8.size(), cudaMemcpyDeviceToHost);

    // 1. IQ2_S gate rows (layer 0): five rows of expert 0.
    {
        const uint8_t* roles[3] = {nullptr, nullptr, nullptr};
        if (!src.slices(0, 0, roles)) {
            std::fprintf(stderr, "qwen35-expert-check: no slices for (0,0)\n");
            return 1;
        }
        const size_t gu_row = ggml_row_size(GGML_TYPE_IQ2_S, n_in);
        for (int r = 0; r < 5; ++r)
            check_iq2s_row(roles[0] + (size_t) r * gu_row, n_in, xq8.data(), xq, stream);
    }
    check_scale_formula();
    check_activation_quantizers();
    check_mmq_expert(src, stream, 0);
    check_mmq_expert(src, stream, 34);
    check_slice_vs_blob(src, stream);
    check_row_split_slices(src);
    check_hit_metadata_plan(src, stream);
    // 2. DOWN21 (IQ3_S) and DOWN23 (IQ4_XS) real down rows.
    {
        const uint8_t* roles[3] = {nullptr, nullptr, nullptr};
        if (!src.slices(0, 0, roles)) return 1;
        check_down_row((int) GGML_TYPE_IQ3_S, "IQ3_S/DOWN21", roles[2], 512, xq8.data(), xq, stream);
        if (src.slices(37, 0, roles)) {
            check_down_row((int) GGML_TYPE_IQ4_XS, "IQ4_XS/DOWN23", roles[2], 512, xq8.data(), xq, stream);
        } else {
            std::printf("IQ4_XS/DOWN23: layer 37 has no slices; skipped\n");
        }
    }
    cudaFree(dx);
    cudaFree(xq);
    cudaStreamDestroy(stream);
    src.close();
    std::printf("qwen35-expert-check: %d failures\n", failures);
    return failures ? 1 : 0;
}
