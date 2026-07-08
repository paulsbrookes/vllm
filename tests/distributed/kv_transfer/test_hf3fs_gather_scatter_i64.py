# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the hf3fs KV offload gather/scatter kernels.

Both kernels address the per-layer offload cache with `token_idx * hidden_size`
in int32, where token_idx comes from an int32 token-index tensor. The offload
cache is sized beyond GPU memory, so token_idx * hidden crosses 2**31 well
within its design envelope. These tests scatter to / gather from a single high
token index in a >2**31-element cache and check the copied row.

Each test allocates ~2.15 GiB of GPU memory (int8 cache).
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.gather_scatter_helper import (  # noqa: E501
    gather_kv_caches,
    scatter_kv_caches,
)
from vllm.platforms import current_platform

HIDDEN = 1024
# (TOTAL_TOKENS - 1) * HIDDEN must exceed 2**31.
TOTAL_TOKENS = 2_100_000


def _pattern(device: torch.device) -> torch.Tensor:
    return ((torch.arange(HIDDEN, device=device) % 127) - 63).to(torch.int8)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_gather_kv_caches_i64_indexing():
    device = torch.device("cuda:0")
    target_token = TOTAL_TOKENS - 1  # token_idx * hidden overflows int32 here

    cache = torch.zeros(TOTAL_TOKENS, HIDDEN, dtype=torch.int8, device=device)
    pattern = _pattern(device)
    cache[target_token] = pattern

    kv_caches_ptrs = torch.tensor([cache.data_ptr()], dtype=torch.int64, device=device)
    dst = torch.empty(1, 1, HIDDEN, dtype=torch.int8, device=device)

    gather_kv_caches(kv_caches_ptrs, TOTAL_TOKENS, dst, [target_token], is_mla=True)

    assert torch.equal(dst[0, 0], pattern)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_scatter_kv_caches_i64_indexing():
    device = torch.device("cuda:0")
    target_token = TOTAL_TOKENS - 1  # token_idx * hidden overflows int32 here

    cache = torch.zeros(TOTAL_TOKENS, HIDDEN, dtype=torch.int8, device=device)
    pattern = _pattern(device)
    src = pattern.reshape(1, 1, HIDDEN).contiguous()

    kv_caches_ptrs = torch.tensor([cache.data_ptr()], dtype=torch.int64, device=device)

    scatter_kv_caches(kv_caches_ptrs, TOTAL_TOKENS, src, [target_token], is_mla=True)

    assert torch.equal(cache[target_token], pattern)
