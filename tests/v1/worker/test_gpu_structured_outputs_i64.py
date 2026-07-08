# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the grammar-bitmask structured-outputs kernel.

_apply_grammar_bitmask_kernel stores into logits at `logits_idx * logits_stride`
in int32, where logits_idx is loaded from an int32 index tensor. Spec-decode
expands the logits rows, so with a large vocab `logits_idx * vocab` crosses
2**31 and the -inf stores land out of bounds. This maps one all-zero bitmask
(mask everything) to the last logits row and checks that row becomes -inf.

Allocates ~4.3 GiB of GPU memory (the logits tensor).
"""

import pytest
import torch
import triton

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

VOCAB_SIZE = 262144
# Last row offset (NUM_ROWS - 1) * VOCAB_SIZE must exceed 2**31.
NUM_ROWS = 8200


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_apply_grammar_bitmask_i64_indexing():
    device = torch.device("cuda:0")
    logits = torch.zeros(NUM_ROWS, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
    target_row = NUM_ROWS - 1  # logits_idx * logits_stride overflows int32 here

    # One mask row mapped to the last logits row; an all-zero bitmask masks
    # every token, so the whole target row must become -inf.
    logits_indices = torch.tensor([target_row], dtype=torch.int32, device=device)
    bitmask = torch.zeros(1, VOCAB_SIZE // 32, dtype=torch.int32, device=device)

    BLOCK_SIZE = 8192
    grid = (1, triton.cdiv(VOCAB_SIZE, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    assert torch.isinf(logits[target_row]).all()
    assert logits[target_row].max() < 0  # -inf, not +inf
    assert torch.equal(logits[0], torch.zeros_like(logits[0]))
