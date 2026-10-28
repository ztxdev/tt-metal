# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue #62903. Run only with the layout guards installed: old code may corrupt DRAM."""
import math

import pytest
import torch
import ttnn


@pytest.fixture
def layout_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    device.cache_entries_counter.reset()
    try:
        yield
    finally:
        device.disable_and_clear_program_cache()


def _tensor(device, value, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        value.to(torch.bfloat16), layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "maximum", "eq", "add_scalar"])
@pytest.mark.parametrize("shape", [(1, 1, 64, 96), (1, 1, 33, 40)])
@pytest.mark.parametrize("warm", [False, True])
def test_binary_preallocated_layout(device, layout_cache, expect_error, name, shape, warm):
    a_host = (torch.arange(math.prod(shape)).reshape(shape) % 7).to(torch.bfloat16)
    b_host = torch.full(shape, 2, dtype=torch.bfloat16)
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    op = ttnn.add if name == "add_scalar" else getattr(ttnn, name)
    rhs = 2.0 if name == "add_scalar" else b
    golden = getattr(torch, "add" if name == "add_scalar" else name)(a_host, b_host).to(torch.bfloat16)
    if warm:
        for repeat in range(2):
            good = _tensor(device, torch.zeros(shape))
            before = device.cache_entries_counter.total
            with device.cache_entries_counter.measure():
                result = op(a, rhs, output_tensor=good)
            if repeat:
                assert device.cache_entries_counter.total == before
            assert result.buffer_address() == good.buffer_address()
            assert torch.equal(ttnn.to_torch(good), golden)
    bad = _tensor(device, torch.full(shape, -7), ttnn.ROW_MAJOR_LAYOUT)
    # Sentinel is a secondary check; allocator adjacency is not assumed.
    sentinel = _tensor(device, torch.full((1, 1, 64, 128), 11), ttnn.ROW_MAJOR_LAYOUT)
    before_bad, before_sentinel = ttnn.to_torch(bad).clone(), ttnn.to_torch(sentinel).clone()
    for _ in range(2):
        with expect_error(RuntimeError, "must match input tensor A layout"):
            op(a, rhs, output_tensor=bad)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(bad), before_bad)
    assert torch.equal(ttnn.to_torch(sentinel), before_sentinel)
