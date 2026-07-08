# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in rejection_random_sample_kernel.

The kernel loads target_probs (and draft_probs) at
`(start_idx + pos) * vocab_size + draft_token_id` in int32. With a large vocab
the per-token row offset exceeds 2**31 once (start_idx + pos) * vocab_size does,
so the token index must be promoted to int64. This drives a single request with
enough accepted draft tokens that the sequential loop reaches a row past 2**31.

Allocates ~4.3 GiB of GPU memory (target_probs).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.rejection_sampler import rejection_random_sample_kernel

VOCAB_SIZE = 262144
# Row offset (NUM_DRAFT - 1) * VOCAB_SIZE must exceed 2**31.
NUM_DRAFT = 8200
BONUS_TOKEN_ID = 999


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_rejection_random_sample_i64_indexing():
    device = torch.device("cuda:0")
    max_spec_len = NUM_DRAFT

    # All draft tokens are id 0; uniform_prob 0 and target_prob 0 => every token
    # is accepted, so the loop runs to the last (overflowing) row and appends
    # the bonus token.
    target_probs = torch.zeros(
        NUM_DRAFT, VOCAB_SIZE, dtype=torch.bfloat16, device=device
    )
    draft_token_ids = torch.zeros(NUM_DRAFT, dtype=torch.int64, device=device)
    recovered_token_ids = torch.zeros(NUM_DRAFT, dtype=torch.int64, device=device)
    uniform_probs = torch.zeros(NUM_DRAFT, dtype=torch.float32, device=device)
    cu_num_draft_tokens = torch.tensor([NUM_DRAFT], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([BONUS_TOKEN_ID], dtype=torch.int64, device=device)
    is_greedy = torch.zeros(1, dtype=torch.bool, device=device)
    output_token_ids = torch.full(
        (1, max_spec_len + 1), -1, dtype=torch.int64, device=device
    )

    rejection_random_sample_kernel[(1,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        None,  # draft_probs (unused: NO_DRAFT_PROBS)
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        VOCAB_SIZE,
        None,  # synthetic_conditional_rates (unused)
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    # Last accepted draft token (row offset > 2**31) and the appended bonus.
    assert output_token_ids[0, NUM_DRAFT - 1].item() == 0
    assert output_token_ids[0, NUM_DRAFT].item() == BONUS_TOKEN_ID
