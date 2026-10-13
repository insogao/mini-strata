// include/strata/core/qwen35_mtp.hpp - task 232: the qwen35moe MTP draft layer's REAL weight reader and
// strict schema validation (single dense layer, blk.<n_layers>, 20 tensors).  Default OFF: nothing here runs
// unless STRATA_QWEN35_MTP=1; the runtime draft/verify/commit machinery is NOT implemented - the probe only
// proves the file's draft block exists with the required shapes/types and fails closed otherwise.
#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace strata::core {

/// One required tensor of the draft layer; shapes are GGUF order (dim 0 varies fastest).  A shape entry of 0
/// means "this dimension is not checked" (used only where the family's width is not fixed by the 229 evidence).
struct Qwen35DraftTensor {
    const char* suffix;   ///< name after "blk.<n_layers>."
    int ndim;             ///< required number of dimensions (1, 2 or 3)
    uint64_t shape[3];    ///< checked shape (0 = unchecked)
    uint32_t types[4];    ///< allowed ggml type ids (GGML_TYPE_F32 is 0, a REAL id: count via n_types)
    int n_types;          ///< valid entries in types
    bool required;
};

/// The fixed 20-tensor draft schema (task 229 evidence: 20/20 present; GU Q2_K, DOWN Q4_K, eh_proj Q8_0
/// [4096,2048]; norms F32; routers BF16; the draft shares the main model's head - no separate head weight).
const std::vector<Qwen35DraftTensor>& qwen35_mtp_schema();

struct Qwen35DraftCheck {
    std::string suffix;
    bool ok = false;
    std::string reason;
};

struct Qwen35DraftReport {
    bool ok = false;
    int checked = 0, missing = 0, mismatched = 0;
    std::vector<Qwen35DraftCheck> checks;
    std::string error;
};

/// Pure schema validation of actual tensors (suffix-less names use the full "blk.<n>.<suffix>" form).
/// Rejects missing tensors, wrong ndim, wrong checked shape, and types outside each tensor's allowed set.
/// Extra tensors are ignored (a real file has more than the draft block).  Never fabricates a draft.
Qwen35DraftReport qwen35_mtp_validate(
    const std::vector<std::tuple<std::string, uint32_t, std::vector<uint64_t>>>& actual, int64_t n_layers);

/// Opens the GGUF read-only (header/metadata/bounds only - tensor payload is never read) and validates the
/// draft layer.  Any open/parse error is a failed report, never an exception escape.
Qwen35DraftReport qwen35_mtp_probe(const std::string& gguf_path, int64_t n_layers);

/// Synthetic contract self-test (no model file, no GPU): 16 named cases over the pure validator.
int qwen35_mtp_selftest();

/// Exact default-OFF gates: STRATA_QWEN35_MTP=1 enables the probe; STRATA_QWEN35_MTP_SELFTEST=1 runs the
/// synthetic contracts.  Anything else (unset/0/other) leaves the existing path untouched.
bool qwen35_mtp_enabled();
bool qwen35_mtp_selftest_requested();

/// Task 243: exact STRATA_QWEN35_MTP_DRAFT=1 gate for the one-round real draft diagnostic (default off).
bool qwen35_mtp_draft_enabled();

}  // namespace strata::core
