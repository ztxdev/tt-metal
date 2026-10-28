# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue #62903. Run only with the layout guards installed: old code may corrupt DRAM."""
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


@pytest.mark.parametrize("name", ["matmul", "linear", "linear_bias", "linear_broadcast_bias"])
@pytest.mark.parametrize("m,k,n", [(64, 96, 64), (33, 64, 40)])
@pytest.mark.parametrize("warm", [False, True])
def test_matmul_preallocated_layout(device, layout_cache, expect_error, name, m, k, n, warm):
    a_host = (1 + torch.arange(m) % 2).reshape(1, 1, m, 1).expand(1, 1, m, k).float()
    # Keep biased outputs below 256 so BF16 rounding differences cannot mask the layout check.
    b_host = (1 + torch.arange(n) % 2).reshape(1, 1, 1, n).expand(1, 1, k, n).float() * 0.5
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    shape = (2 if name == "linear_broadcast_bias" else 1, 1, m, n)
    kwargs = {}
    if name in ("linear_bias", "linear_broadcast_bias"):
        kwargs["bias"] = _tensor(device, torch.ones((shape[0], 1, 1, n)))
    op = ttnn.matmul if name == "matmul" else ttnn.linear
    expected = ((a_host @ b_host) + int("bias" in kwargs)).expand(shape).to(torch.bfloat16)
    if warm:
        for repeat in range(2):
            good = _tensor(device, torch.zeros(shape))
            before = device.cache_entries_counter.total
            with device.cache_entries_counter.measure():
                result = op(a, b, optional_output_tensor=good, **kwargs)
            if repeat:
                assert device.cache_entries_counter.total == before
            assert result.buffer_address() == good.buffer_address()
            assert torch.equal(ttnn.to_torch(good), expected)
    bad = _tensor(device, torch.full(shape, -7), ttnn.ROW_MAJOR_LAYOUT)
    # A broadcast that expands the matmul output is implemented by a trailing binary add.
    error = (
        "must match input tensor A layout"
        if name == "linear_broadcast_bias"
        else "Layout mismatch between optional output tensor"
    )
    with expect_error(RuntimeError, error):
        op(a, b, optional_output_tensor=bad, **kwargs)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(bad), torch.full(shape, -7, dtype=torch.bfloat16))


@pytest.mark.parametrize("untilize", [False, True])
@pytest.mark.parametrize("row_major", [False, True])
def test_matmul_untilize_preallocated_layout(device, layout_cache, expect_error, untilize, row_major):
    shape = (1, 1, 32, 32)
    a_host = (1 + torch.arange(32) % 2).reshape(1, 1, 32, 1).expand(shape).float()
    b_host = (1 + torch.arange(32) % 2).reshape(1, 1, 1, 32).expand(shape).float()
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    expected = (a_host @ b_host).to(torch.bfloat16)
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        mcast_in0=True,
        untilize_out=untilize,
    )
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    # Mcast1D's untilized output is supported by its globally allocated sharded CB.
    # Interleaved untilize uses a tile writer and is characterized separately.
    memory_config = ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.WIDTH)
    out = ttnn.from_torch(
        torch.full(shape, -7, dtype=torch.bfloat16), layout=layout, device=device, memory_config=memory_config
    )
    for _ in range(2):
        if untilize != row_major:
            with expect_error(RuntimeError, "Layout mismatch between optional output tensor"):
                ttnn.matmul(a, b, dtype=ttnn.bfloat16, program_config=config, optional_output_tensor=out)
            assert torch.equal(ttnn.to_torch(out), torch.full(shape, -7, dtype=torch.bfloat16))
        else:
            result = ttnn.matmul(a, b, dtype=ttnn.bfloat16, program_config=config, optional_output_tensor=out)
            assert result.buffer_address() == out.buffer_address()
            assert result.layout == layout
            assert torch.equal(ttnn.to_torch(out), expected)


def _untilize_config():
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        mcast_in0=True,
        untilize_out=True,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("preallocated", [False, True])
def test_matmul_untilize_rejects_interleaved_output(device, layout_cache, expect_error, dtype, preallocated):
    shape = (1, 1, 32, 32)
    a, b = _tensor(device, torch.ones(shape)), _tensor(device, torch.ones(shape))
    kwargs = dict(dtype=dtype, program_config=_untilize_config(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = None
    if preallocated:
        out = ttnn.from_torch(
            torch.full(shape, -7.0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kwargs["optional_output_tensor"] = out
    for _ in range(2):
        before = device.num_program_cache_entries()
        with expect_error(RuntimeError, "untilize_out requires sharded output; interleaved output is not supported"):
            ttnn.matmul(a, b, **kwargs)
        assert device.num_program_cache_entries() == before
        if out is not None:
            ttnn.synchronize_device(device)
            assert torch.equal(ttnn.to_torch(out).float(), torch.full(shape, -7.0))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("preallocated", [False, True])
def test_matmul_untilize_sharded_output(device, layout_cache, dtype, preallocated):
    shape = (1, 1, 32, 32)
    a_host = (1 + torch.arange(32) % 2).reshape(1, 1, 32, 1).expand(shape).float()
    b_host = (1 + torch.arange(32) % 2).reshape(1, 1, 1, 32).expand(shape).float()
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    memory_config = ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.WIDTH)
    kwargs = dict(dtype=dtype, program_config=_untilize_config(), memory_config=memory_config)
    out = None
    if preallocated:
        out = ttnn.from_torch(
            torch.full(shape, -7.0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        kwargs["optional_output_tensor"] = out
    for repeat in range(2):
        before = device.cache_entries_counter.total
        with device.cache_entries_counter.measure():
            result = ttnn.matmul(a, b, **kwargs)
        if repeat:
            assert device.cache_entries_counter.total == before
        assert result.layout == ttnn.ROW_MAJOR_LAYOUT
        if out is not None:
            assert result.buffer_address() == out.buffer_address()
        assert torch.equal(ttnn.to_torch(result).float(), a_host @ b_host)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_matmul_interleaved_explicit_untilize(device, dtype):
    shape = (1, 1, 32, 32)
    a_host = (1 + torch.arange(32) % 2).reshape(1, 1, 32, 1).expand(shape).float()
    b_host = (1 + torch.arange(32) % 2).reshape(1, 1, 1, 32).expand(shape).float()
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    config = _untilize_config()
    config.untilize_out = False
    tiled = ttnn.matmul(a, b, dtype=dtype, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    row_major = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
    assert row_major.layout == ttnn.ROW_MAJOR_LAYOUT
    assert torch.equal(ttnn.to_torch(row_major).float(), a_host @ b_host)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_linear_untilize_config_keeps_tile_output(device, layout_cache, expect_error, dtype, with_bias, sharded):
    # linear currently sets MatmulParams.untilize_out=false, independently of the
    # Mcast1D config field. Preserve that API behavior rather than enabling a new mode.
    shape = (1, 1, 32, 32)
    a_host = (1 + torch.arange(32) % 2).reshape(1, 1, 32, 1).expand(shape).float()
    b_host = (1 + torch.arange(32) % 2).reshape(1, 1, 1, 32).expand(shape).float()
    a, b = _tensor(device, a_host), _tensor(device, b_host)
    memory_config = (
        ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.WIDTH)
        if sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = dict(dtype=dtype, program_config=_untilize_config(), memory_config=memory_config)
    if with_bias:
        kwargs["bias"] = _tensor(device, torch.ones((1, 1, 1, 32)))
    expected = a_host @ b_host + int(with_bias)
    good = ttnn.from_torch(
        torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    for output in (None, good, good):
        result = ttnn.linear(a, b, optional_output_tensor=output, **kwargs)
        assert result.layout == ttnn.TILE_LAYOUT
        if output is not None:
            assert result.buffer_address() == output.buffer_address()
        assert torch.equal(ttnn.to_torch(result).float(), expected)
    bad = ttnn.from_torch(
        torch.full(shape, -7.0), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    before = device.num_program_cache_entries()
    with expect_error(RuntimeError, "Layout mismatch between optional output tensor"):
        ttnn.linear(a, b, optional_output_tensor=bad, **kwargs)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == before
    assert torch.equal(ttnn.to_torch(bad).float(), torch.full(shape, -7.0))
