# IAI GEMM tiled-layout sweep

This experiment changes only the physical `layout_tiled` tile extents used by
the GEMM benchmark. It does not change IAI execution, schedule selection, or
cache budgets. Raw nanobench CSVs are under
`results/tile-sweep/tigerlake-i5-1135g7-4c-base-2.4ghz/`.

## Controls

- Release build with `CMAKE_CXX_FLAGS=-march=native`.
- Four OpenMP threads pinned to Linux CPUs 0, 1, 2, and 3 with
  `OMP_PLACES='{0},{1},{2},{3}'`.
- Four distinct physical Tiger Lake cores at fixed 2.4 GHz under the
  `performance` governor.
- `TI`, `TJ`, and `TK` instantiate A as `(TI,TK)`, B as `(TK,TJ)`, and C as
  `(TI,TJ)`. Every inner layout remains right-major.
- Automatic `iai::md_reduce` execution was measured. The 64x32x64 winner also
  passed the checked DT, TV, automatic, forced-complete, forced-blocked, and
  BLAS correctness paths at N=128.

The screen covered 32 fragment-aligned shapes. Small output tiles intended to
admit 2x2 physical-tile blocks were consistently slower: additional tile and
fragment traversal outweighed their modeled reuse. Deepening `TK` was more
effective because the current executor retains each register fragment for one
native reduction tile.

## Finalists

| Tiles `TI`x`TJ`x`TK` | 1024 time | vs 32-cubed | 2048 time | vs 32-cubed | 4096 time | vs 32-cubed mean |
|:---------------------|----------:|------------:|----------:|------------:|----------:|-----------------:|
| 32x32x32 | 0.043711 s | baseline | 0.420447 s | baseline | 3.463775 s | baseline |
| 64x16x64 | 0.036117 s | -17.4% | 0.375057 s | -10.8% | 3.163129 s | -8.7% |
| 64x16x128 | 0.037254 s | -14.8% | 0.344631 s | -18.0% | 3.114497 s | -10.1% |
| 32x16x128 | 0.039782 s | -9.0% | 0.370087 s | -12.0% | 3.013624 s | -13.0% |
| 64x32x64 | 0.038900 s | -11.0% | 0.384725 s | -8.5% | 2.786436 s | -19.6% |

The two 4096 control runs were 3.617780 and 3.309770 seconds. The table uses
their 3.463775-second mean. Therefore the 64x32x64 improvement is 15.8% versus
the faster control and 23.0% versus the slower control, so it remains larger
than the observed run drift.

At 4096, 64x32x64 changes calling-worker counters relative to the mean 32-cubed
control as follows:

| Metric | 32-cubed mean | 64x32x64 | Change |
|:-------|--------------:|----------:|-------:|
| Time | 3.463775 s | 2.786436 s | -19.6% |
| Instructions | 8.881B | 8.187B | -7.8% |
| Branches | 313.4M | 290.0M | -7.5% |
| L1D misses | 167.0M | 157.3M | -5.8% |
| LLC references | 26.5M | 15.7M | -40.6% |
| LLC misses | 15.5M | 8.8M | -43.2% |

Fresh four-thread BLAS took 1.614768 seconds. The tile change narrows IAI's
same-session ratio to BLAS from 2.15x for the mean 32-cubed control to 1.73x.

## Interpretation

For 64x32x64, the A, B, and C native tiles occupy 32 KiB, 16 KiB, and 16 KiB,
respectively. The 64 KiB unit footprint exceeds the current 32 KiB cache-block
budget, so the unchanged policy falls back to a unit physical-tile block. The
gain therefore comes from storage geometry and within-tile execution rather
than a new IAI block-selection decision:

- `TK=64` halves the number of native reduction-tile boundaries relative to
  32-cubed, so C fragments are loaded and stored less frequently;
- `TI=64` increases reuse of each B tile across output rows;
- `TJ=32` retains two complete 16-column register fragments and avoids the
  poor behavior observed for very narrow or very wide J tiles; and
- the resulting balance materially reduces upper-cache traffic at 4096.

No single shape wins every size. 64x16x64 is best at 1024, 64x16x128 is best
at 2048, and 64x32x64 is best at 4096. If one static shape is required for the
tested range, 64x32x64 is the strongest large-problem choice and improves on
32-cubed at every measured size. It is therefore the new benchmark default;
the three dimensions remain CMake-configurable for further experiments.

## Small output tiles and deep reduction tiles

A follow-up screen tested the hypothesis that smaller `TI` and `TJ`, paired
with a much larger `TK`, would improve register reuse. It did not hold for the
current physical-tile executor:

| Tiles `TI`x`TJ`x`TK` | 1024 time | vs control | L1D misses | vs control |
|:---------------------|----------:|-----------:|------------:|-----------:|
| 16x16x128 | 0.052781 s | +20.8% | 4.49M | +93% |
| 16x16x256 | 0.053853 s | +37.5% | 8.58M | +256% |
| 16x16x512 | 0.052047 s | +32.9% | 10.80M | +348% |
| 8x16x128 | 0.075196 s | +72.0% | 10.71M | +361% |
| 4x16x128 | 0.123888 s | +216.3% | 10.77M | +347% |

The 128-depth rows use the original 32-cubed run as their same-session
control. The later 256- and 512-depth rows use the mean of bracketing 32-cubed
controls. Those controls drifted from 0.031363 to 0.046974 seconds, but even
the fastest control does not change the ordering above.

Keeping `TI` at least 32 changes the outcome, but the benefit stops before an
arbitrarily deep K tile:

| Tiles `TI`x`TJ`x`TK` | 1024 time | vs 32-cubed | 2048 time | vs 32-cubed |
|:---------------------|----------:|-------------:|----------:|-------------:|
| 32x16x64 | 0.041802 s | -4.4% | -- | -- |
| 32x16x128 | 0.039782 s | -9.0% | 0.370087 s | -12.0% |
| 32x16x256 | 0.039135 s | -10.5% | 0.404611 s | -3.8% |
| 32x16x512 | 0.041797 s | +6.7% | -- | -- |
| 64x16x64 | 0.036117 s | -17.4% | 0.375057 s | -10.8% |
| 64x16x128 | 0.037254 s | -14.8% | 0.344631 s | -18.0% |

For 32x16, increasing `TK` from 32 to 128 removes reduction-tile boundaries
and lowers instruction overhead. At `TK=256` and 512, however, L1D misses grow
to roughly 2.1x and 4.5x their respective controls. The 256 result consequently
loses most of its advantage at 2048, while 512 is already slower at 1024.
Shrinking `TI` also creates more output work items and reloads each B region
for more row tiles. Thus a narrower J tile is useful here only when paired with
an unchanged or larger I tile; making both output dimensions small is not.

## Relation to high-performance GEMM designs

This result agrees with the central distinction in Goto-style GEMM and BLIS:
a cache-resident `mc` by `kc` macro-panel is not the same object as the small
`mr` by `nr` register micro-kernel. See
[Goto and van de Geijn's GEMM anatomy](https://valeriyvan.com/assets/docs/Anatomy%20of%20High-Performance%20Matrix%20Multiplication.pdf)
and the
[BLIS framework paper](https://www.cs.utexas.edu/~flame/pubs/blis_toms_rev1.pdf).
Making the physical storage tile micro-kernel-sized collapses those two levels
and exposes tile traversal and panel-reload costs.

State-of-the-art GPU GEMMs retain the same multilevel idea despite a different
memory and execution model. CUTLASS independently tiles at threadblock, warp,
thread, and instruction levels; its K mainloop is staged and double-buffered
rather than represented by one unbounded K tile. Larger M/N tiles improve reuse
but trade against available parallelism. See NVIDIA's
[efficient GEMM description](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).
The CUTLASS 3/CuTe API similarly composes shared-memory layouts, tiled copy
operations, and tiled MMA atoms instead of assigning one shape every role; see
the
[CUTLASS 3 GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md).
Finally, cache-aware output-tile ordering is itself valuable: PyTorch's
[persistent grouped-GEMM work](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
improves L2 reuse by grouping the visitation order of output tiles.

The transferable design lesson for IAI is therefore not "use small M/N and a
huge K." It is to keep the native tiled layout, cache-block schedule, and
register fragment as separately composable levels. A future executor can group
multiple native tiles for cross-tile cache reuse, choose a bounded K stage,
and order output work to retain an A or B region, without forcing the physical
storage tile to serve as the register tile or cache block.
