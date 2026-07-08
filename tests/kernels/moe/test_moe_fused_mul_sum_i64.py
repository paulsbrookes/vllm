# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the moe_fused_mul_sum kernel.

The kernel indexes the (num_tokens, top_k, hidden) expert-output workspace with
`offs_m * stride_m` where `stride_m = top_k * hidden`, computed in int32. With a
DeepSeek-scale row stride (top_k=8, hidden=7168 -> 57344) the last token's base
offset exceeds 2**31 once num_tokens * top_k * hidden > 2**31, so offs_m must be
promoted to int64. This checks the last token's fused sum against a reference.

Allocates ~4.3 GiB of GPU memory (the inputs workspace).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.platforms import current_platform

TOP_K = 8
HIDDEN = 7168
# (NUM_TOKENS - 1) * TOP_K * HIDDEN must exceed 2**31.
NUM_TOKENS = 37500


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_moe_fused_mul_sum_i64_indexing():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    inputs = torch.randn(NUM_TOKENS, TOP_K, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_weights = torch.randn(NUM_TOKENS, TOP_K, dtype=torch.bfloat16, device=device)

    # Reference for the last token, whose offs_m * stride_m overflows int32.
    ref_last = (inputs[-1].float() * topk_weights[-1].float().unsqueeze(-1)).sum(0)

    out = moe_fused_mul_sum(inputs, topk_weights)

    torch.testing.assert_close(out[-1].float(), ref_last, rtol=1e-2, atol=1e-2)
