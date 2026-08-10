#!/usr/bin/env python3
"""Simplest possible conv2d test kernel for the Ascend backend.

Usage:
    python test_conv2d_simple.py

This test verifies that al.conv2d compiles and runs correctly on Ascend NPU.
It compares the result against torch.nn.functional.conv2d as reference.
"""

import os
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


# --- Kernel 1: minimal unbatched conv2d (no bias) ---


@triton.jit
def conv2d_kernel_3d(
    input_ptr,
    weight_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
):
    """
    Simple 3D conv2d: input (C, H, W), weight (C_out, C, kH, kW) -> output (C_out, H_out, W_out)

    No tiling — loads the entire tensor at once.
    """
    H_out: tl.constexpr = H + 2 * pad_h - kH + 1
    W_out: tl.constexpr = W + 2 * pad_w - kW + 1

    # Load input tile: shape (C, H, W)
    c_offs = tl.arange(0, C)[:, None, None]
    h_offs = tl.arange(0, H)[None, :, None]
    w_offs = tl.arange(0, W)[None, None, :]
    input_offs = c_offs * H * W + h_offs * W + w_offs
    input_tile = tl.load(input_ptr + input_offs)

    # Load weight tile: shape (C_out, C, kH, kW)
    co_offs = tl.arange(0, C_out)[:, None, None, None]
    ci_offs = tl.arange(0, C)[None, :, None, None]
    kh_offs = tl.arange(0, kH)[None, None, :, None]
    kw_offs = tl.arange(0, kW)[None, None, None, :]
    weight_offs = co_offs * C * kH * kW + ci_offs * kH * kW + kh_offs * kW + kw_offs
    weight_tile = tl.load(weight_ptr + weight_offs)

    # 2D convolution
    output = al.conv2d(
        input_tile, weight_tile,
        stride=(1, 1), padding=(pad_h, pad_w), dilation=(1, 1), groups=1,
    )

    # Store output: shape (C_out, H_out, W_out)
    lout_h = tl.arange(0, H_out)[None, :, None]
    lout_w = tl.arange(0, W_out)[None, None, :]
    co_offs_2d = tl.arange(0, C_out)[:, None, None]
    out_offs = co_offs_2d * H_out * W_out + lout_h * W_out + lout_w
    tl.store(output_ptr + out_offs, output)


# --- Kernel 2: batched conv2d (with bias) ---


@triton.jit
def conv2d_kernel_4d(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
):
    """
    Batched 4D conv2d: input (N, C, H, W), weight (C_out, C, kH, kW) -> output (N, C_out, H_out, W_out)
    """
    H_out: tl.constexpr = H - kH + 1
    W_out: tl.constexpr = W - kW + 1

    # Load input tile: shape (N, C, H, W)
    n_offs = tl.arange(0, N)[:, None, None, None]
    c_offs = tl.arange(0, C)[None, :, None, None]
    h_offs = tl.arange(0, H)[None, None, :, None]
    w_offs = tl.arange(0, W)[None, None, None, :]
    input_offs = n_offs * C * H * W + c_offs * H * W + h_offs * W + w_offs
    input_tile = tl.load(input_ptr + input_offs)

    # Load weight tile: shape (C_out, C, kH, kW)
    co_offs = tl.arange(0, C_out)[:, None, None, None]
    ci_offs = tl.arange(0, C)[None, :, None, None]
    kh_offs = tl.arange(0, kH)[None, None, :, None]
    kw_offs = tl.arange(0, kW)[None, None, None, :]
    weight_offs = co_offs * C * kH * kW + ci_offs * kH * kW + kh_offs * kW + kw_offs
    weight_tile = tl.load(weight_ptr + weight_offs)

    # Load bias: shape (C_out,)
    bias_offs = tl.arange(0, C_out)
    bias_tile = tl.load(bias_ptr + bias_offs)

    # 2D convolution
    output = al.conv2d(
        input_tile, weight_tile, bias_tile,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
    )

    # Store output: shape (N, C_out, H_out, W_out)
    n_offs2 = tl.arange(0, N)[:, None, None, None]
    co_offs2 = tl.arange(0, C_out)[None, :, None, None]
    hout_offs = tl.arange(0, H_out)[None, None, :, None]
    wout_offs = tl.arange(0, W_out)[None, None, None, :]
    out_offs = n_offs2 * C_out * H_out * W_out + co_offs2 * H_out * W_out + hout_offs * W_out + wout_offs
    tl.store(output_ptr + out_offs, output)


def test_conv2d_3d():
    """Test unbatched conv2d: (C=4, H=8, W=8) with kernel (C_out=8, C_in=4, kH=3, kW=3) -> (8, 6, 6)"""
    C, H, W = 4, 8, 8
    C_out, kH, kW = 8, 3, 3
    pad_h, pad_w = 0, 0
    H_out, W_out = H - kH + 1, W - kW + 1

    print(f"Test 3D: input ({C}, {H}, {W}), weight ({C_out}, {C}, {kH}, {kW}) -> output ({C_out}, {H_out}, {W_out})")

    input_t = torch.randn(C, H, W, dtype=torch.float16, device="npu")
    weight_t = torch.randn(C_out, C, kH, kW, dtype=torch.float16, device="npu")
    output_t = torch.empty(C_out, H_out, W_out, dtype=torch.float16, device="npu")

    conv2d_kernel_3d[(1,)](input_t, weight_t, output_t,
                           C=C, H=H, W=W, C_out=C_out, kH=kH, kW=kW,
                           pad_h=pad_h, pad_w=pad_w)
    torch.npu.synchronize()

    ref = torch.nn.functional.conv2d(
        input_t.unsqueeze(0).float(),
        weight_t.float(), bias=None,
        stride=1, padding=0, dilation=1, groups=1,
    ).squeeze(0).half()

    max_err = (output_t.float() - ref.float()).abs().max().item()
    print(f"  Max error: {max_err:.6f}")
    assert max_err < 0.1, f"Error too large: {max_err}"
    print("  PASSED ✓")


def test_conv2d_4d():
    """Test batched conv2d: (N=2, C=4, H=8, W=8) with kernel (C_out=8, C_in=4, kH=3, kW=3) -> (2, 8, 6, 6)"""
    N, C, H, W = 2, 4, 8, 8
    C_out, kH, kW = 8, 3, 3
    H_out, W_out = H - kH + 1, W - kW + 1

    print(f"Test 4D: input ({N}, {C}, {H}, {W}), weight ({C_out}, {C}, {kH}, {kW}) -> output ({N}, {C_out}, {H_out}, {W_out})")

    input_t = torch.randn(N, C, H, W, dtype=torch.float16, device="npu")
    weight_t = torch.randn(C_out, C, kH, kW, dtype=torch.float16, device="npu")
    bias_t = torch.randn(C_out, dtype=torch.float16, device="npu")
    output_t = torch.empty(N, C_out, H_out, W_out, dtype=torch.float16, device="npu")

    conv2d_kernel_4d[(1,)](input_t, weight_t, bias_t, output_t,
                           N=N, C=C, H=H, W=W, C_out=C_out, kH=kH, kW=kW)
    torch.npu.synchronize()

    ref = torch.nn.functional.conv2d(
        input_t.float(), weight_t.float(), bias=bias_t.float(),
        stride=1, padding=0, dilation=1, groups=1,
    ).half()

    max_err = (output_t.float() - ref.float()).abs().max().item()
    print(f"  Max error: {max_err:.6f}")
    assert max_err < 0.1, f"Error too large: {max_err}"
    print("  PASSED ✓")


if __name__ == "__main__":
    test_conv2d_3d()
    test_conv2d_4d()
    print("\nAll tests passed!")
