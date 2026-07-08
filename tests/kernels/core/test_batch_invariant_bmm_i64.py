# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing for the batch base pointer in the batch-invariant bmm kernel.

bmm_kernel already promotes the in-tile offs_m/offs_n/offs_k to int64 under the
A_LARGE/B_LARGE/C_LARGE guard, but the per-batch base pointers
`pid_b * stride_ab/bb/cb` stay int32. With enough batches the last batch's base
offset `(B - 1) * M * N` exceeds int32, so pid_b must also be promoted. This
test uses many batches (small M/K) so C is > 2**31 elements and checks the last
batch matches a reference.

Allocates ~4.3 GiB of GPU memory (the C output).
"""

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import bmm_batch_invariant
from vllm.platforms import current_platform

# (B - 1) * M * N must exceed 2**31 while each batch stays < 2**31.
B, M, N, K = 2050, 1024, 1024, 8


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_bmm_batch_invariant_i64_batch_offset():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    a = torch.randn(B, M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)

    # Reference for the last batch, where pid_b * stride_cb overflows int32.
    ref_last = (a[-1].float() @ b[-1].float()).to(torch.bfloat16)

    out = bmm_batch_invariant(a, b)

    torch.testing.assert_close(out[-1], ref_last, rtol=1e-2, atol=1e-2)
