# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the _num_nans logits kernel.

_num_nans_kernel offsets logits with `req_idx * logits_stride`. With a large
vocab the last row's base offset exceeds int32 (8192 * 262144 == 2**31), so
`req_idx` must be promoted to int64 before the stride multiply, matching the
sibling sampler kernels (gumbel.py, min_p.py). This test plants a known NaN
count in the last row and checks the kernel counts it correctly.

Allocates ~4.3 GiB of GPU memory.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.metrics.logits import get_num_nans

VOCAB_SIZE = 262144  # 2**18, GLM/Gemma-scale
# Last row offset (NUM_REQS - 1) * VOCAB_SIZE must exceed 2**31.
NUM_REQS = 8200


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_num_nans_i64_indexing():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    logits = torch.zeros(NUM_REQS, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
    # Distinct, known NaN counts in the first (no overflow) and last
    # (offset > 2**31) rows.
    logits[0, :3] = float("nan")
    logits[-1, :7] = float("nan")

    num_nans = get_num_nans(logits)

    assert num_nans[0].item() == 3
    assert num_nans[-1].item() == 7
    assert num_nans[1:-1].sum().item() == 0
