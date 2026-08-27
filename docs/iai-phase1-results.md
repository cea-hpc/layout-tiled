# IAI Phase 1 results

The Phase 1 executor measurements and the follow-up physical-tile sweep were
collected on 2026-08-03. Raw CSVs, process logs, topology, affinity, and
frequency metadata are under
`results/phase1/tigerlake-i5-1135g7-4c-base-2.4ghz/` and
`results/tile-sweep/tigerlake-i5-1135g7-4c-base-2.4ghz/`.

## Controls and names

- The Release build used `CMAKE_CXX_FLAGS=-march=native`.
- OpenMP and OpenBLAS used four threads on Linux CPUs 0, 1, 2, and 3. These
  are four distinct Tiger Lake physical cores (hwloc L#0, L#2, L#4, L#6).
- `OMP_PLACES='{0},{1},{2},{3}'` and `taskset --cpu-list 0,1,2,3` prevented
  migration and SMT oversubscription.
- Every selected CPU used the `performance` governor with min=max=base=2.4
  GHz.
- Nanobench counters cover the calling worker only, not the complete
  four-thread team. Their consistent scope supports relative comparisons.
- The controlled executor comparison used 32x32x32 GEMM tiles. The subsequent
  layout sweep selected 64x32x64 as the final GEMM benchmark default.

`IAI-AUTO`, `IAI-COMPLETE`, and `IAI-BLOCKED` mean automatic schedule
selection, forced complete-reduction fragments, and forced
reduction-blocked fragments. `TRRR` is the existing GEMM/contraction shorthand
for tiled A, B, and C operands whose inner layouts are right-major. GEMV's
legacy `TRL` instead abbreviates its matrix mapping directly: tiled storage,
right-major outer tile grid, left-major inner tile. `L1`, `L2`, `RECT`, and
`BATCH1` identify tile-shape presets; they do not assert measured cache
residency.

The optimized forced paths passed four-thread correctness checks for GEMM,
GEMV, and the compatible 32-cubed tensor case. Every timed process exited
successfully. The optimized GEMM binary contains vector FMA instructions.

## GEMM

The first table isolates the Phase 1 executor changes: every tiled
implementation uses the original 32x32x32 physical layout. Times are
wall-clock seconds per invocation from that controlled run.

| N | BLAS | DT | TV | IAI auto | IAI complete | IAI blocked |
|---:|-----:|---:|---:|---------:|-------------:|------------:|
| 512 | 0.003257 | 0.013362 | 0.015528 | 0.006286 | 0.005434 | 0.005720 |
| 1024 | 0.030685 | 0.118539 | 0.120584 | 0.050798 | 0.055065 | 0.053576 |
| 2048 | 0.236198 | 0.975114 | 1.076027 | 0.467955 | 0.611626 | 0.579080 |
| 4096 | 2.091734 | 8.707492 | 11.096303 | 4.974997 | 11.283792 | 5.137092 |

The automatic policy selects blocked execution at every tested GEMM size:
the complete footprint is `8 KiB + 16 KiB * (N / 32)`, which first exceeds
the current 256 KiB complete-reduction budget at N=512. At 4096, automatic
and forced-blocked timings agree closely, while forced-complete is 2.20 times
slower than forced-blocked.

The normalized calling-worker counters from the same 32x32x32 controlled run
are:

| N | Implementation | I/F | L1D misses/kF | LLC refs/kF | LLC misses/kF |
|---:|:---------------|----:|--------------:|------------:|--------------:|
| 512 | BLAS | 0.0306 | 2.764 | 0.047 | 0.017 |
| 512 | DT | 0.1756 | 1.183 | 0.254 | 0.006 |
| 512 | IAI auto | 0.0651 | 1.141 | 0.106 | 0.003 |
| 1024 | BLAS | 0.0292 | 2.735 | 0.022 | 0.014 |
| 1024 | DT | 0.1748 | 1.276 | 0.268 | 0.071 |
| 1024 | IAI auto | 0.0648 | 1.185 | 0.131 | 0.050 |
| 2048 | BLAS | 0.0292 | 2.509 | 0.012 | 0.009 |
| 2048 | DT | 0.1744 | 1.300 | 0.297 | 0.194 |
| 2048 | IAI auto | 0.0647 | 1.245 | 0.167 | 0.074 |
| 4096 | BLAS | 0.0288 | 2.432 | 0.006 | 0.005 |
| 4096 | DT | 0.1741 | 1.293 | 0.289 | 0.156 |
| 4096 | IAI auto | 0.0646 | 1.218 | 0.187 | 0.109 |

Relative to the starting `layout_tiled::reduce`, automatic `iai::md_reduce`
changes the normalized cache events as follows:

| N | Old L1D/kF | New L1D/kF | Change | Old LLC refs/kF | New LLC refs/kF | Change | Old LLC misses/kF | New LLC misses/kF | Change |
|---:|-----------:|-----------:|-------:|----------------:|----------------:|-------:|------------------:|------------------:|-------:|
| 512 | 5.916 | 1.141 | -80.7% | 0.141 | 0.106 | -24.8% | 0.000 | 0.003 | noise-dominated |
| 1024 | 5.911 | 1.185 | -79.9% | 0.172 | 0.131 | -23.9% | 0.018 | 0.050 | +171.0% |
| 2048 | 5.915 | 1.245 | -78.9% | 0.286 | 0.167 | -41.7% | 0.107 | 0.074 | -31.1% |
| 4096 | 5.572 | 1.218 | -78.1% | 1.341 | 0.187 | -86.1% | 0.180 | 0.109 | -39.5% |

Absolute old-to-new times are not directly usable: despite matching recorded
controls, every freshly rerun BLAS/DT/TV control was 1.5 to 3.0 times slower
than its baseline counterpart while retiring essentially the same instruction
count. This indicates a systemic run-to-run timing shift. Same-run ratios are
more defensible. At N=4096, the old reduce took 0.723 times the baseline DT
time; automatic IAI takes 0.571 times the final DT time, a 21.0% relative
improvement. Its ratio to BLAS improves from 2.634 to 2.378, or 9.7%.

At N=512 the policy boundary is slightly too aggressive: forced-complete is
about 5% faster than forced-blocked. This is evidence for the later portable
cache-policy tuning phase, not for embedding a kernel-specific exception.

### Final GEMM benchmark default

After the executor comparison, a physical tile-shape sweep selected 64x32x64
as the final benchmark default. It improves automatic `iai::md_reduce` at all
three remeasured sizes without changing IAI execution or policy:

| N | 32x32x32 control | 64x32x64 final | Change |
|---:|-----------------:|----------------:|-------:|
| 1024 | 0.043711 s | 0.038900 s | -11.0% |
| 2048 | 0.420447 s | 0.384725 s | -8.5% |
| 4096 | 3.463775 s | 2.786436 s | -19.6% |

The 4096 control is the mean of bracketing runs at 3.617780 and 3.309770
seconds. The final layout is therefore 15.8% faster than the faster control
and 23.0% faster than the slower one, so the improvement exceeds the observed
run drift. A fresh same-session four-thread BLAS run took 1.614768 seconds:
64x32x64 narrows IAI's ratio to BLAS from 2.15x to 1.73x.

At 4096, its calling-worker counters change as follows:

| Metric | 32x32x32 mean | 64x32x64 final | Change |
|:-------|--------------:|----------------:|-------:|
| Instructions | 8.881B | 8.187B | -7.8% |
| Branches | 313.4M | 290.0M | -7.5% |
| L1D misses | 167.0M | 157.3M | -5.8% |
| LLC references | 26.5M | 15.7M | -40.6% |
| LLC misses | 15.5M | 8.8M | -43.2% |

The 64x32x64 configuration passed the checked DT, TV, automatic,
forced-complete, forced-blocked, and BLAS paths at N=128. N=512 was not
remeasured in the tile sweep. Full candidates, controls, and interpretation
are recorded in `docs/iai-gemm-tile-sweep.md`.

## GEMV

| N | BLAS | DT | TV | IAI auto | IAI complete | IAI blocked | Automatic path |
|---:|-----:|---:|---:|---------:|-------------:|------------:|:---------------|
| 1024 | 0.000436 | 0.000437 | 0.000517 | 0.000573 | 0.000438 | 0.000421 | complete |
| 4096 | 0.010220 | 0.010206 | 0.011873 | 0.019672 | 0.011306 | 0.042657 | blocked |
| 8192 | 0.042081 | 0.041789 | 0.047123 | 0.042470 | 0.041954 | 0.050939 | blocked |
| 16384 | 0.161343 | 0.167158 | 0.162705 | 0.163556 | 0.161745 | 0.162940 | blocked |

GEMV is bandwidth-bound at the large sizes and all implementations converge.
The automatic and forced-blocked 4096 timings show unexplained variance even
though their selected block shape and counter profiles agree; this point
should be repeated before using it to tune policy. At 8192 and 16384,
automatic IAI is within 2% of BLAS and DT/TV.

## Tensor contraction

Automatic IAI remains faster than the corresponding DT implementation for
every tested shape:

| N | Tile | DT | IAI auto | Speedup over DT | Automatic path |
|---:|:-----|---:|---------:|----------------:|:---------------|
| 32 | L1 | 0.000098 | 0.000048 | 2.05x | complete |
| 64 | L1 | 0.002183 | 0.000977 | 2.23x | complete |
| 64 | L2 | 0.002077 | 0.000852 | 2.44x | complete |
| 64 | RECT | 0.002074 | 0.000947 | 2.19x | complete |
| 64 | BATCH1 | 0.002206 | 0.000960 | 2.30x | complete |
| 128 | L1 | 0.036150 | 0.013772 | 2.62x | complete |
| 128 | L2 | 0.031519 | 0.015180 | 2.08x | blocked |
| 128 | RECT | 0.032595 | 0.015477 | 2.11x | blocked |
| 128 | BATCH1 | 0.031566 | 0.013338 | 2.37x | complete |
| 256 | L1 | 0.606705 | 0.269477 | 2.25x | complete |
| 256 | L2 | 0.532188 | 0.232909 | 2.28x | blocked |
| 256 | RECT | 0.528329 | 0.310055 | 1.70x | blocked |
| 256 | BATCH1 | 0.487646 | 0.239980 | 2.03x | complete |

At 256 cubed, blocking reduces calling-worker L1 misses from 5.429 to 1.565
per kF for the L2 tile shape and from 5.589 to 1.860 per kF for RECT. The
automatic path selects blocked for both shapes. Blocked execution is not
universally faster: for the smaller-footprint L1 and BATCH1 shapes, automatic
correctly retains complete execution even though forcing blocked reduces L1
misses. This demonstrates why cache events and elapsed time must both inform a
future portable policy.

## Phase 1 conclusion

The cache-blocked native-tile schedule materially removes the original GEMM
cache-traffic pathology without giving up tiled spatial locality. Selecting
64x32x64 as the final GEMM layout then improves the large case by another
19.6% and nearly halves its LLC misses. Together, these results support the
design premise: native tiled storage supplies spatial locality, while the IAI
schedule supplies temporal reuse. Phase 1 is complete; cache-budget discovery,
calibration, and policy injection remain intentionally deferred to the
dedicated later phase.
