#!/usr/bin/env python3
"""Minimal ROCm portability patch for SplaTAM's pinned depth rasterizer.

PyTorch CUDAExtension performs the CUDA->HIP source conversion. Several
classes of CUDA-only constructs survive that conversion unchanged and must
be neutralized here, all guarded so plain-CUDA builds (the "a100cluster"
reference platform) are byte-for-byte unaffected:

1. cooperative_groups/reduce.h — included by forward.cu/backward.cu/
   rasterizer_impl.cu but cg::reduce is never actually called; ROCm does not
   ship this header. (The original version of this patch only covered
   forward.cu/backward.cu and missed rasterizer_impl.cu, which includes it
   too -- confirmed by a real ROCm build failing on
   "rasterizer_impl.hip:29:10: fatal error: 'cooperative_groups/reduce.h'
   file not found" even after guarding the other two files.)
2. device_launch_parameters.h — included by forward.h/backward.h/
   rasterizer_impl.cu for CUDA builtins (threadIdx, blockDim, ...) that are
   already available via hip/hip_runtime.h once hipify_python rewrites
   cuda_runtime.h; torch's CUDA_INCLUDE_MAP does not rewrite or drop this
   particular header, so it survives hipification as a dangling include and
   ROCm has no such file (fatal error: 'device_launch_parameters.h' file not
   found), breaking every downstream build.
3. Spaced kernel-launch chevron syntax ("<< <grid, block> >>" instead of
   "<<<grid, block>>>") — used throughout forward.cu/backward.cu/
   rasterizer_impl.cu. hipify_python's launch-syntax regex expects the
   contiguous "<<<...>>>" form; the spaced form passes through unconverted
   and hipcc then parses "<<" as a raw left-shift operator, producing
   "error: expected expression" at every kernel launch site. Normalize to
   the contiguous form unconditionally (harmless under nvcc too).
4. __trap() — a CUDA-only device intrinsic used once in auxiliary.h to abort
   a broken kernel invocation. HIP has no __trap(); the portable equivalent
   is __builtin_trap(), which both nvcc and hipcc/clang support, so this
   substitution does not need a backend guard.
"""
from pathlib import Path

COOPERATIVE_GROUPS_FILES = (
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "cuda_rasterizer/rasterizer_impl.cu",
)
DEVICE_LAUNCH_PARAMS_FILES = (
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/backward.h",
    "cuda_rasterizer/rasterizer_impl.cu",
)
CHEVRON_FILES = (
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "cuda_rasterizer/rasterizer_impl.cu",
)
TRAP_FILES = ("cuda_rasterizer/auxiliary.h",)


def guard_include(path: Path, needle: str) -> None:
    text = path.read_text()
    if needle not in text:
        raise SystemExit(f"expected include not found in {path}; pinned source drifted")
    guarded = (
        '#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n'
        f'{needle}\n'
        '#endif'
    )
    path.write_text(text.replace(needle, guarded, 1))


def normalize_chevrons(path: Path) -> None:
    text = path.read_text()
    patched = text.replace("<< <", "<<<").replace(">> >", ">>>")
    if patched != text:
        path.write_text(patched)


def replace_trap(path: Path) -> None:
    text = path.read_text()
    needle = "__trap();"
    if needle not in text:
        raise SystemExit(f"expected __trap() call not found in {path}; pinned source drifted")
    path.write_text(text.replace(needle, "__builtin_trap();", 1))


for name in COOPERATIVE_GROUPS_FILES:
    guard_include(Path(name), '#include <cooperative_groups/reduce.h>')

for name in DEVICE_LAUNCH_PARAMS_FILES:
    guard_include(Path(name), '#include "device_launch_parameters.h"')

for name in CHEVRON_FILES:
    normalize_chevrons(Path(name))

for name in TRAP_FILES:
    replace_trap(Path(name))
