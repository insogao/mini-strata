# MTP runtime prototype: external review entry point

This branch publishes the OpenCode implementation used by the M268 Windows/CUDA experiments. It is an experimental review snapshot, not a production merge or a claim of stable acceleration. PR #1 supplies the earlier Decode baseline; this review adds the real MTP runtime and its independent candidates. The frozen Prefill production executable/model remain separate and unchanged.

Start with [the current scheme list](CURRENT_SCHEMES_FOR_DISCUSSION_20261013.md), [measured evidence](../../evidence/mtp-runtime-20261013.json), and [source identity](../../evidence/mtp-review-source-20261013.json). Historical updates in the scheme list retain their dates; the latest completed runtime snapshot here is M268, not the pending P271 small-T expert prototype.

## Source map

| Area | Code to review | Questions |
|---|---|---|
| Actual draft payload/schema and experimental model | `src/core/qwen35_mtp.cpp`, `include/strata/core/qwen35_mtp.hpp`, `src/program/qwen35_model.cpp` | Twenty blk.40 tensors, supported types/shapes, model identity, draft allocation |
| Greedy K=1 loop | `src/core/qwen35_mtp_txn.cpp`, `include/strata/core/qwen35_mtp_txn.hpp` | Seed/draft/bonus ownership, target two-row outputs, hidden provenance, acceptance and EOS |
| Draft state catch-up | `qwen35_mtp_txn.cpp`: accepted-token replay | Real token plus target row0 hidden; independent draft KV; unused replay logits |
| Serial verifier candidate | `qwen35_mtp_txn.cpp`: `STRATA_QWEN35_MTP_SERIAL_VERIFY` | Immediate h0 copy before another decode; h1 is row0 of the second call; rejection has no speculative target row |
| Actual T1/T2 expert routing | `src/core/qwen35_bridge.cpp`: `seam_exec`, `src/core/expert_source.cpp`: `expert_native_decode_layer`, `src/core/qwen35_prefill.cu` | T1 native service versus T2 prompt/MMQ pipeline, GPU/CPU waits, staging lifetimes |
| GDN and rollback implementation | `vendor/llama.cpp/src/models/delta-net-base.cpp`, `vendor/llama.cpp/src/models/qwen35moe.cpp`, recurrent/hybrid memory files | GDN_AR versus GDN_CH, snapshots, accepted prefix, rejection and context-tail state |
| Compact expert slots | `src/core/expert_cache.cpp`, `include/strata/core/expert_cache.hpp`, `src/core/expert_source.cpp` | Per-layer blob sizes, rank-to-slot layout, span checks, Prefill borrowing/restoration |
| GPU type compatibility | `CMakeLists.txt`, `src/kernels/cuda/iq_kernels.cu`, `src/prefill/moe_mmq.cu` | MMQ/K-quants/Q6 support; actual Q6 DOWN geometry |
| Actual output timing | `qwen35_model.cpp`, `qwen35_mtp_txn.cpp` | Typed post-fflush timestamps, per-request anchors, JSON-visible versus EOS-complete time |

The required target bonus logits are not the unused head mentioned in the scheme discussion: the unused output belongs to accepted-token **draft replay**. `decode_rows(..., all_outputs=false)` still requests the last row's logits. Simply deleting replay would violate the current independent draft-KV contract.

## What was measured

RTX4060 Laptop 8GiB, Windows, t8/c4096/ub4096, new UD-IQ3_S MTP GGUF, 2700 compact slots, fixed 3896-token benign input. Both arms use real MTP ON/K1; only serial target verification differs. Each process handles two complete requests, clearing and re-prefilling between them.

| Second request | T2 target control | Serial T1 target candidate |
|---|---:|---:|
| Prefill | 2.992089 s | 3.000233 s |
| Actual first-token flush TTFT | 3.000668 s | 3.009228 s |
| Complete JSON visible | 3.727006 s | 3.651867 s |
| EOS complete | 3.727243 s | 3.652200 s |
| Decode `(n-1)/(last-first flush)` | 20.6515 tok/s | 23.3413 tok/s |

All four requests produced the same complete benign JSON and EOS, with 16 visible tokens and all eight rounds accepted. This is a single paired screen. The previous T2 control in M267 was 22.931 tok/s / EOS 3.566 s, so the apparent M268 difference is not a proven stable speedup. Cold first-request Prefill differed markedly and is retained in the evidence.

Actual routing counters support the source-path distinction: T2 arm `decode_layers=0, prefill_layers=720`; serial arm `decode_layers=1280, prefill_layers=80`. Serial verification also switches the target GDN kernel form. This experiment therefore does not isolate MoE execution from GDN and graph scheduling.

Compact caching separately saved 1,400,823,808 bytes of arena capacity at 2700 slots. Peak whole-card VRAM was sampled at four points, not continuously traced: M268 peaks were 7586.7 and 7616.0 MiB. These samples do not establish a guaranteed memory margin.

## Build and runtime configuration

Use an x64 Visual Studio developer shell with CMake/Ninja and CUDA 12.8. The local tested build used Release, sm_89 and the checked-in `vendor/llama.cpp` model implementation:

```powershell
cmake -S . -B build-mtp-review -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_QWEN35=ON -DSTRATA_QWEN35_TOOLS=ON `
  -DSTRATA_GGML_DIR="$PWD/vendor/llama.cpp" `
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON `
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-mtp-review --target strata
```

The recipe describes the tested build settings; the publication checkout was not rebuilt during upload. CUDA was tested locally. HIP/SYCL and other GPUs were not validated. The source manifest records exact copied file hashes; normal Git line-ending conversion can change checkout byte hashes without changing the source text. No executable is shipped.

Relevant environment flags:

```text
STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1
STRATA_QWEN35_PACKED_IDS=1
STRATA_QWEN35_EXPERT_ORDER=count_desc
STRATA_QWEN35_COMPACT_SLOTS=1
STRATA_QWEN35_MTP_TXN=1
STRATA_QWEN35_MTP_TXN_K=1
STRATA_QWEN35_MTP_TXN_REQUESTS=2
STRATA_QWEN35_STOP_EOG=1
STRATA_QWEN35_DECODE_TIMESTAMPS=1
STRATA_QWEN35_MTP_SERIAL_VERIFY=0 or 1
STRATA_QWEN35_MTP_COST_SPLIT=0
STRATA_QWEN35_MTP_COST_SYNC=0
```

Check the actual environment names in the source before running; provide your own model-matching native expert pack/profile and a legal tokenized input. The experimental model SHA is in the evidence. The older production GGUF has no blk.40 draft weights. New and old GGUF main tensors have 49 quantization-type changes, so historical production performance is not a same-model MTP OFF control.

The public snapshot intentionally excludes business prompts/token arrays, models, binaries, credentials and OpenCode chat logs. It supports source review and building with your own inputs; it cannot exactly reproduce the private business fixture from the repository alone. The tested Windows GPU job wrapper is included as `tools/gpu-lock.py`: its job/lock/input paths must be supplied for your own environment; it does not make separate GPU jobs safe to time concurrently.

## Open gates and requested feedback

- Risk labels, real rejection/rollback, context-tail cases and 128–256-token explanations are not accepted business tests yet.
- Serial verification writes recurrent snapshots differently; the current serial branch does not consume rollback slots. Any later rollback use needs a new contract proof.
- The corrected M268 cost boundaries were not profiled again in this screen. Earlier M267 windows overlapped; their exact phase percentages and clamped residual are not authoritative.
- M268's first build attempt failed in its launcher before compiling; the second incremental build succeeded. Its static checker ran 18 cases despite an eight-case budget. Failures and budget deviations are retained, not treated as additional performance evidence.
- MTP loops, compact slots and serial verification are opt-in. No production defaults have been adopted based on these results.

Please prioritize concrete proposals for the small-T target path, increasing real draft length, eliminating unnecessary draft replay outputs, or safely fusing accepted replay with the next draft. For each proposal specify the function/dependency, numerical and state contract, likely saving, minimum implementation, fair comparison, and bounded exit condition.
