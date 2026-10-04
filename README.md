# Sigma-Delta Motion Detection — Scalar vs NEON SIMD

Motion detection pipeline for ARM/AArch64 targets: Sigma-Delta background
subtraction + morphological erosion/dilation. Each stage has a scalar C
implementation and an ARM NEON SIMD implementation, selected at compile time.

## Pipeline

| Stage | Function | Description |
|---|---|---|
| Background subtraction | `Sigma_Delta` | Per-pixel adaptive mean/variance model; classifies each pixel as background (`WHITE`) or motion (`BLACK`) |
| Erosion | `Erosion` | 3×3 neighborhood, output `BLACK` only if all 9 neighbors are `BLACK` |
| Dilatation | `Dilatation` | 3×3 neighborhood, output `BLACK` if any neighbor is `BLACK` |

## NEON intrinsics used

| Intrinsic | Operation |
|---|---|
| `vld1q_u8` / `vld1_u8` | Load 16 / 8 bytes from memory |
| `vst1q_u8` / `vst1_u8` | Store 16 / 8 bytes to memory |
| `vdupq_n_u8` / `vdup_n_u8` | Broadcast a scalar to all lanes |
| `vandq_u8` / `vorrq_u8` | Bitwise AND / OR across all lanes |
| `vadd_u8` | Lane-wise add (8-bit) |
| `vqadd_u8` / `vqsub_u8` | Saturating add / subtract |
| `vmax_u8` / `vmin_u8` | Lane-wise max / min |
| `vsub_u8` | Lane-wise subtract |
| `vshr_n_u8` | Right shift by immediate, all lanes |
| `vclt_u8` / `vcgt_u8` | Lane-wise compare (`<`, `>`), produces a mask |
| `vbsl_u8` | Bitwise select between two vectors using a mask |
| `vextq_u8` | Byte-shifted window across two 128-bit vectors (used to derive neighbor offsets from aligned loads) |

`_u8` variants operate on `uint8x8_t` (8 lanes); `q_u8` variants operate on
`uint8x16_t` (16 lanes).

## Build

Three builds for a 3-way comparison: pure scalar, compiler auto-vectorized, and
hand-written NEON intrinsics.

```bash
# 1. Scalar — no vectorization
gcc main.c lib_bmp.c -I. -O3 -fno-tree-vectorize -o main_scalar.exe

# 2. Scalar source, GCC auto-vectorization enabled (NEON is baseline on AArch64,
#    so -O3 alone lets GCC vectorize the plain loops on its own)
gcc main.c lib_bmp.c -I. -O3 -o main_scalar_autovec.exe

# 3. Hand-written NEON intrinsics
gcc main.c lib_bmp.c -I. -O3 -DSIMD_VERSION -o main_simd.exe
```

| Build | `SIMD_VERSION` | `-fno-tree-vectorize` | Code path |
|---|:---:|:---:|---|
| `main_scalar.exe` | off | yes | Plain C, one pixel/lane at a time, no SIMD |
| `main_scalar_autovec.exe` | off | no | Plain C, GCC auto-vectorizes to NEON where it can |
| `main_simd.exe` | on | n/a | Hand-written NEON intrinsics |

## Run

```bash
./main_scalar.exe
./main_scalar_autovec.exe
./main_simd.exe
```

Reads `images_src/Image{0..199}.bmp`, writes results to `images_simd/` or
`images_scalaire/`, prints average per-stage processing time in ns.

## Benchmark

MediaTek Helio G99 (2× Cortex-A76 @ 2.2 GHz, 6× Cortex-A55) Phone's processor.
Values are medians over multiple runs.

| Stage | Scalar (ns/image) | Auto-vectorized (ns/image) | NEON intrinsics (ns/image) | Speedup (scalar → NEON) |
|---|---:|---:|---:|---:|
| Sigma/Delta | 2 288 735.98 | 85 945.50 | 54 915.33 | 41.7x |
| Erosion | 436 344.39 | 150 603.88 | 22 380.97 | 19.5x |
| Dilatation | 1 399 405.01 | 359 169.96 | 23 187.34 | 60.4x |

## Results analysis

- **NEON intrinsics outperform both other builds on every stage**, with the
  largest margin on Dilatation (60.4x over scalar, 15.5x over auto-vectorized).
- **Auto-vectorization helps substantially but leaves a clear gap to hand-written
  NEON**, and that gap is not uniform across stages:
  - Sigma/Delta: auto-vec is already close to hand-tuned NEON (1.6x apart). The
    stage is mostly straight-line arithmetic and simple comparisons — a pattern
    GCC's vectorizer handles well on its own.
  - Erosion: 2.9x gap. Moderate — the `&&`-chained neighborhood check is harder
    to auto-vectorize than pure arithmetic, but still partly tractable.
  - Dilatation: 3.9x gap, the largest. The `||`-chained neighborhood check rarely
    short-circuits on typical frames (see pipeline description above), so the
    compiler ends up auto-vectorizing a loop that still does near-maximal work
    per pixel — hand-written NEON avoids this entirely by computing all 9
    neighbor comparisons as unconditional, branch-free vector ops.
- **Takeaway**: compiler auto-vectorization is a reasonable default for simple,
  branch-light kernels, but branch-heavy per-pixel logic (as in the morphological
  filters here) is where hand-written SIMD intrinsics provide the most value over
  relying on `-O3` alone.

## Structure

```
.
├── main.c
├── lib_bmp.h
├── lib_bmp.c
├── images_src/
├── images_simd/
├── images_scalaire/
└── README.md
```

## Requirements

- AArch64 (ARM64) target
- GCC with C11 support
- POSIX environment
