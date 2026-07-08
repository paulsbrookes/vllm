# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in vllm.v1.sample.rejection_sampler kernels.

rejection_random_sample_kernel and sample_recovered_tokens_kernel address the
[num_tokens, vocab_size] draft/target probability tensors with an int32
`token_idx * vocab_size` stride multiply (token_idx = start_idx + pos, where
start_idx/pos come from tl.load/tl.program_id). With a GLM-scale vocab
(~155k) that product exceeds int32 once token_idx >= ~13.8k -- reachable via
max_num_seqs * (1 + num_speculative_tokens) in a large speculative batch,
exactly as confirmed end-to-end against a real model in
vllm-a1-repros/repro_h1_end2end.py (see A1-int64-kernel-offsets.md, finding
H1/H2).

Each test places one "request under test" at a high token index within a
large filler batch (driven by a big per-request draft-token count K, so the
*request* count stays small) and checks its output matches the same request
run alone at token index 0. The `req_idx * vocab_size` stride in
sample_recovered_tokens_kernel's inv_q read is fixed by the same patch but
not independently regression-tested here: reaching it requires >~13.8k
*concurrent requests* in one batch (as opposed to a large per-request K),
which needs a second ~8.6 GiB fp32 buffer and is far less reachable in
practice than the token_idx path exercised below.

Peak allocation ~13 GiB (bf16/fp32 [14000, 155264] buffers).
"""

import pytest
import torch

from tests.utils import large_gpu_test
from tests.v1.sample.test_rejection_sampler import (
    DEVICE_TYPE,
    create_logits_tensor,
    create_sampling_metadata,
)
from vllm.platforms import current_platform
from vllm.v1.sample.rejection_sampler import rejection_sample, sample_recovered_tokens

VOCAB_SIZE = 155264
THRESHOLD = -(-(2**31) // VOCAB_SIZE)  # ceil(2**31 / VOCAB_SIZE)
K = 140
NUM_REQS = 100
ROI_REQ = NUM_REQS - 1
ROI_START = ROI_REQ * K

assert ROI_START > THRESHOLD, "test config must land past the overflow row"


def _run_recovered(
    target_probs_roi: torch.Tensor,  # [K, VOCAB_SIZE]
    draft_token_ids_roi: list[int],  # length K
    num_reqs: int,
    roi_req: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run sample_recovered_tokens on a batch of num_reqs requests (each
    with K draft tokens), with the request under test at roi_req. Returns
    its K recovered-token-id rows."""
    num_tokens = num_reqs * K
    roi_start = roi_req * K

    target_probs = torch.zeros(
        num_tokens, VOCAB_SIZE, dtype=torch.float32, device=device
    )
    target_probs[roi_start : roi_start + K] = target_probs_roi

    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    draft_token_ids[roi_start : roi_start + K] = torch.tensor(
        draft_token_ids_roi, dtype=torch.int32, device=device
    )

    cu_num_draft_tokens = torch.arange(
        K, num_tokens + 1, K, dtype=torch.int32, device=device
    )

    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=torch.ones(num_reqs, device=device),
        generators={roi_req: torch.Generator(device=device).manual_seed(seed)},
    )

    recovered = sample_recovered_tokens(
        max_spec_len=K,
        num_draft_tokens=[K] * num_reqs,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_token_ids=draft_token_ids,
        draft_probs=None,
        target_probs=target_probs,
        sampling_metadata=sampling_metadata,
        device=device,
    )
    return recovered[roi_start : roi_start + K].clone()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@large_gpu_test(min_gb=20)
def test_sample_recovered_tokens_i64_offset():
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)

    target_probs_roi = torch.softmax(torch.randn(K, VOCAB_SIZE, device=device), dim=-1)
    draft_token_ids_roi = [(100 + i) % VOCAB_SIZE for i in range(K)]

    ref = _run_recovered(
        target_probs_roi,
        draft_token_ids_roi,
        num_reqs=1,
        roi_req=0,
        seed=0,
        device=device,
    )
    big = _run_recovered(
        target_probs_roi,
        draft_token_ids_roi,
        num_reqs=NUM_REQS,
        roi_req=ROI_REQ,
        seed=0,
        device=device,
    )
    assert torch.equal(big, ref)


def _run_rejection_sample(
    target_logits_roi: torch.Tensor,  # [K, VOCAB_SIZE]
    draft_token_ids_roi: list[int],  # length K
    bonus_token_id_roi: int,
    num_reqs: int,
    roi_req: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run rejection_sample on a batch of num_reqs requests (each with K
    draft tokens), with the request under test at roi_req. Returns its
    output row (length K + 1, including the bonus token slot)."""
    num_tokens = num_reqs * K
    roi_start = roi_req * K

    target_logits = torch.full(
        (num_tokens, VOCAB_SIZE), -100.0, dtype=torch.bfloat16, device=device
    )
    target_logits[roi_start : roi_start + K] = target_logits_roi.to(torch.bfloat16)

    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    draft_token_ids[roi_start : roi_start + K] = torch.tensor(
        draft_token_ids_roi, dtype=torch.int32, device=device
    )

    cu_num_draft_tokens = torch.arange(
        K, num_tokens + 1, K, dtype=torch.int32, device=device
    )

    bonus_token_ids = torch.zeros(num_reqs, 1, dtype=torch.int32, device=device)
    bonus_token_ids[roi_req, 0] = bonus_token_id_roi

    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=torch.ones(num_reqs, device=device),
        generators={roi_req: torch.Generator(device=device).manual_seed(seed)},
    )

    output = rejection_sample(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[K] * num_reqs,
        max_spec_len=K,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
    )
    return output[roi_req].clone()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@large_gpu_test(min_gb=20)
def test_rejection_sample_i64_offset():
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)

    # Huge logit margin -> target_prob ~= 1.0 for the draft token at every
    # position, so every draft token is accepted and the per-request loop
    # walks all K overflowing rows instead of stopping at the first one.
    draft_token_ids_roi = [200 + i for i in range(K)]
    bonus_token_id_roi = 99999
    target_logits_roi = create_logits_tensor(
        [draft_token_ids_roi + [bonus_token_id_roi]], vocab_size=VOCAB_SIZE
    )

    ref = _run_rejection_sample(
        target_logits_roi,
        draft_token_ids_roi,
        bonus_token_id_roi,
        num_reqs=1,
        roi_req=0,
        seed=0,
        device=device,
    )
    big = _run_rejection_sample(
        target_logits_roi,
        draft_token_ids_roi,
        bonus_token_id_roi,
        num_reqs=NUM_REQS,
        roi_req=ROI_REQ,
        seed=0,
        device=device,
    )
    assert torch.equal(big, ref)

    expected = torch.tensor(
        draft_token_ids_roi + [bonus_token_id_roi], dtype=torch.int32, device=device
    )
    assert torch.equal(ref, expected)
