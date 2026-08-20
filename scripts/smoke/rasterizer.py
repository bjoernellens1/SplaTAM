#!/usr/bin/env python3
"""Tiny dependency and forward/backward smoke for the SplaTAM image."""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

expected = os.environ.get("EXPECT_BACKEND", "").strip().lower()
assert expected in {"rocm", "cuda"}, "EXPECT_BACKEND must be rocm or cuda"
assert torch.cuda.is_available(), "GPU backend is not visible to torch"
if expected == "rocm":
    assert torch.version.hip, "ROCm image must expose torch.version.hip"
else:
    assert torch.version.hip is None, "CUDA author image unexpectedly reports a HIP runtime"

# The audited SplaTAM stack uses legacy Open3D 0.16. It is CPU-side in this
# controlled mapper image, but must remain importable alongside the modern ROCm
# PyTorch stack. Open3D <=0.18 requires NumPy <2.
assert o3d.__version__.startswith("0.16."), o3d.__version__
assert int(np.__version__.split(".", 1)[0]) < 2, np.__version__

device = torch.device("cuda:0")  # PyTorch intentionally uses cuda:* for ROCm too.
bg = torch.zeros(3, device=device)
view = torch.eye(4, device=device)
proj = torch.eye(4, device=device)
settings = GaussianRasterizationSettings(
    image_height=16,
    image_width=16,
    tanfovx=1.0,
    tanfovy=1.0,
    bg=bg,
    scale_modifier=1.0,
    viewmatrix=view,
    projmatrix=proj,
    sh_degree=0,
    campos=torch.zeros(3, device=device),
    prefiltered=False,
)
rasterizer = GaussianRasterizer(settings)
means3d = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
means2d = torch.zeros((1, 3), device=device, requires_grad=True)
opacities = torch.tensor([[0.8]], device=device, requires_grad=True)
colors = torch.tensor([[0.25, 0.5, 0.75]], device=device, requires_grad=True)
scales = torch.tensor([[0.05, 0.05, 0.05]], device=device, requires_grad=True)
rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
color, radii, depth = rasterizer(
    means3D=means3d,
    means2D=means2d,
    opacities=opacities,
    colors_precomp=colors,
    scales=scales,
    rotations=rotations,
)
assert color.is_cuda and depth.is_cuda and radii.is_cuda
assert torch.isfinite(color).all() and torch.isfinite(depth).all()
(color.sum() + depth.sum() * 1e-3).backward()
for name, tensor in {
    "means3d": means3d,
    "opacities": opacities,
    "colors": colors,
    "scales": scales,
    "rotations": rotations,
}.items():
    assert tensor.grad is not None, f"missing gradient for {name}"
    assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
print(
    "splatam smoke passed",
    {
        "backend": expected,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "open3d": o3d.__version__,
        "numpy": np.__version__,
    },
)
