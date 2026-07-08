# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in sample_recovered_tokens_kernel.

The kernel loads target/draft probs at `token_idx * vocab_size + vocab_offset`
in int32. With a large vocab `token_idx * vocab_size` crosses 2**31 for high
token indices, so token_idx must be promoted to int64. This places a single
non-zero probability in the last (overflowing) token row and checks the kernel
recovers that token via argmax.

Allocates ~4.3 GiB of GPU memory (target_probs).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.rejection_sampler import sample_recovered_tokens_kernel

VOCAB_SIZE = 262144
# Row offset (NUM_DRAFT - 1) * VOCAB_SIZE must exceed 2**31.
NUM_DRAFT = 8200
TARGET_TOKEN = 12345


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_sample_recovered_tokens_i64_indexing():
    device = torch.device("cuda:0")

    target_probs = torch.zeros(
        NUM_DRAFT, VOCAB_SIZE, dtype=torch.bfloat16, device=device
    )
    # Only the last token row has a winner, at TARGET_TOKEN.
    target_probs[NUM_DRAFT - 1, TARGET_TOKEN] = 1.0
    inv_q = torch.ones(1, VOCAB_SIZE, dtype=torch.float32, device=device)
    draft_token_ids = torch.zeros(NUM_DRAFT, dtype=torch.int64, device=device)
    cu_num_draft_tokens = torch.tensor([NUM_DRAFT], dtype=torch.int32, device=device)
    output_token_ids = torch.full((NUM_DRAFT,), -1, dtype=torch.int64, device=device)

    sample_recovered_tokens_kernel[(1, NUM_DRAFT)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        None,  # draft_probs (unused: NO_DRAFT_PROBS)
        target_probs,
        inv_q,
        VOCAB_SIZE,
        BLOCK_SIZE=8192,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert output_token_ids[NUM_DRAFT - 1].item() == TARGET_TOKEN
