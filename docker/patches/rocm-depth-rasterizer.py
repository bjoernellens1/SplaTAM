#!/usr/bin/env python3
"""Minimal ROCm portability patch for SplaTAM's pinned depth rasterizer.

PyTorch CUDAExtension performs the CUDA->HIP source conversion. The pinned
rasterizer includes cooperative_groups/reduce.h but does not use cg::reduce;
ROCm does not provide that header. Keep CUDA behavior unchanged and omit only
the unused header when compiling under HIP.
"""
from pathlib import Path

for name in ("cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu"):
    path = Path(name)
    text = path.read_text()
    needle = '#include <cooperative_groups/reduce.h>'
    if needle not in text:
        raise SystemExit(f"expected include not found in {name}; pinned source drifted")
    text = text.replace(
        needle,
        '#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n'
        '#include <cooperative_groups/reduce.h>\n'
        '#endif',
        1,
    )
    path.write_text(text)
