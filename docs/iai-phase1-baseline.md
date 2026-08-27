# IAI Phase 1 baseline

This baseline was collected on 2026-08-03 at commit
`1ab04d255ffd7bf51e8621da9bafe2b2a02beb7d` from a dirty working tree. The raw
results are in
`results/baseline/tigerlake-i5-1135g7-4c-base-2.4ghz/`; its `metadata.txt` is
the authoritative record of the machine and run configuration.

## Execution controls

- CPU: Intel Core i5-1135G7, four physical cores and two hardware threads per
  core.
- Frequency: `performance` governor with every policy constrained to
  min=max=base=2.4 GHz. The global turbo capability remained enabled, but the
  2.4 GHz policy ceiling prevented turbo frequencies.
- Placement: Linux CPUs 0, 1, 2, and 3, which are hwloc physical PUs P#0,
  P#1, P#2, and P#3 and occupy distinct physical cores. In hwloc's logical
  namespace these same PUs are L#0, L#2, L#4, and L#6.
- OpenMP: four threads, dynamic teams disabled, close binding, and the quoted
  explicit placement `OMP_PLACES='{0},{1},{2},{3}'`.
- BLAS: OpenMP-enabled OpenBLAS with four threads. Affinity display verified
  that both the benchmark OpenMP regions and OpenBLAS placed one thread on
  each Linux CPU from 0 through 3.
- Isolation: each GEMM and GEMV implementation ran in a fresh process under
  `taskset --cpu-list 0,1,2,3`. Tensor variants ran in one fresh process per
  size. Affinity prevents migration outside the selected CPUs but cannot stop
  the OS from descheduling a worker; `/usr/bin/time -v` context-switch counts
  are retained in `logs/`.

Correctness was checked separately at four threads. GEMM DT, TV, reduce, and
BLAS passed at extent 128; all four GEMV variants passed at extent 128; and the
three compatible L1 tensor variants passed at extent 32. All 11 repository
unit tests passed.

## Counter scope

Times are wall-clock seconds per kernel invocation. Nanobench opens Linux perf
events on the calling thread without inheritance, so the instruction and
cache counters below cover the main OpenMP worker's share, not the sum over
the four-worker team. This consistent scope is sufficient for the relative
Phase 1 comparisons below; normalized counts must not be interpreted as
whole-team aggregate events per FLOP.

The cache columns below are events per 1,000 logical FLOPs. `I/F` is calling-
thread instructions per logical FLOP.

## GEMM

| N | Implementation | Time (s) | I/F | L1D misses / kF | LLC refs / kF | LLC misses / kF |
|---:|:---------------|---------:|----:|----------------:|--------------:|----------------:|
| 512 | BLAS | 0.002116 | 0.0306 | 2.693 | 0.025 | 0.008 |
| 512 | DT | 0.007457 | 0.1756 | 1.056 | 0.132 | 0.001 |
| 512 | TV | 0.006601 | 0.1936 | 1.105 | 0.156 | 0.000 |
| 512 | reduce | 0.002354 | 0.0564 | 5.916 | 0.141 | 0.000 |
| 1024 | BLAS | 0.016073 | 0.0292 | 2.751 | 0.013 | 0.005 |
| 1024 | DT | 0.052727 | 0.1746 | 1.056 | 0.225 | 0.038 |
| 1024 | TV | 0.054463 | 0.1935 | 1.118 | 0.196 | 0.024 |
| 1024 | reduce | 0.020590 | 0.0562 | 5.911 | 0.172 | 0.018 |
| 2048 | BLAS | 0.123360 | 0.0290 | 2.464 | 0.007 | 0.006 |
| 2048 | DT | 0.440434 | 0.1744 | 1.054 | 0.215 | 0.107 |
| 2048 | TV | 0.459378 | 0.1935 | 1.108 | 0.183 | 0.092 |
| 2048 | reduce | 0.197010 | 0.0561 | 5.915 | 0.286 | 0.107 |
| 4096 | BLAS | 0.991143 | 0.0281 | 2.439 | 0.004 | 0.003 |
| 4096 | DT | 3.610129 | 0.1741 | 1.057 | 0.249 | 0.208 |
| 4096 | TV | 3.746622 | 0.1934 | 1.116 | 0.211 | 0.166 |
| 4096 | reduce | 2.610611 | 0.0560 | 5.572 | 1.341 | 0.180 |

At 4096, reduce is 1.38x faster than DT and 1.44x faster than TV, but remains
2.63x slower than BLAS. Its normalized L1D misses are about 5.27x DT's, while
its LLC references are about 5.38x DT's. Its LLC misses are not comparably
worse, which suggests substantially more LLC traffic and reuse rather than a
proportional increase in memory fetches.

## GEMV

| N | Implementation | Time (s) | I/F | L1D misses / kF | LLC refs / kF | LLC misses / kF |
|---:|:---------------|---------:|----:|----------------:|--------------:|----------------:|
| 1024 | BLAS | 0.000129 | 0.0579 | 15.876 | 3.934 | 1.094 |
| 1024 | DT | 0.000086 | 0.1437 | 17.690 | 5.529 | 0.153 |
| 1024 | TV | 0.000132 | 0.1444 | 17.667 | 5.017 | 0.405 |
| 1024 | reduce | 0.000112 | 0.2364 | 17.764 | 0.450 | 0.149 |
| 4096 | BLAS | 0.005888 | 0.0536 | 17.199 | 5.939 | 4.086 |
| 4096 | DT | 0.004097 | 0.1426 | 17.627 | 8.876 | 8.356 |
| 4096 | TV | 0.004159 | 0.1423 | 17.635 | 8.752 | 8.265 |
| 4096 | reduce | 0.004317 | 0.2343 | 17.650 | 5.961 | 5.786 |
| 8192 | BLAS | 0.020019 | 0.0520 | 18.350 | 5.856 | 4.512 |
| 8192 | DT | 0.016504 | 0.1426 | 17.619 | 9.053 | 8.548 |
| 8192 | TV | 0.016740 | 0.1423 | 17.618 | 9.020 | 8.552 |
| 8192 | reduce | 0.016903 | 0.2339 | 17.630 | 5.940 | 5.722 |
| 16384 | BLAS | 0.063441 | 0.0518 | 18.119 | 5.112 | 3.804 |
| 16384 | DT | 0.058957 | 0.1424 | 17.614 | 9.275 | 8.603 |
| 16384 | TV | 0.058171 | 0.1420 | 17.614 | 9.340 | 8.863 |
| 16384 | reduce | 0.059194 | 0.2337 | 17.622 | 6.604 | 6.401 |

At 16384 the four GEMV times are close. Reduce has essentially the same L1D
miss rate as DT/TV and lower LLC traffic, but retires more calling-thread
instructions per FLOP.

## Tensor contraction

Extent 32 runs only the compatible L1 configurations. Extents 64, 128, and 256
also run L2, rectangular, and batch-one configurations; their complete rows
remain in the raw CSVs. The directly comparable L1 rows are:

| N | Implementation | Time (s) | I/F | L1D misses / kF | LLC refs / kF | LLC misses / kF |
|---:|:---------------|---------:|----:|----------------:|--------------:|----------------:|
| 32 | cache-blocked | 0.000063 | 0.2578 | 1.548 | 0.007 | 0.000 |
| 32 | DT | 0.000051 | 0.1985 | 1.512 | 0.006 | 0.000 |
| 32 | reduce | 0.000020 | 0.0708 | 1.512 | 0.007 | 0.000 |
| 64 | cache-blocked | 0.000959 | 0.2547 | 3.864 | 0.082 | 0.000 |
| 64 | DT | 0.000782 | 0.1956 | 2.667 | 0.013 | 0.000 |
| 64 | reduce | 0.000281 | 0.0648 | 2.544 | 0.066 | 0.000 |
| 128 | cache-blocked | 0.015741 | 0.2533 | 4.503 | 0.039 | 0.032 |
| 128 | DT | 0.012553 | 0.1944 | 2.626 | 0.026 | 0.023 |
| 128 | reduce | 0.005141 | 0.0620 | 3.231 | 0.045 | 0.031 |
| 256 | cache-blocked | 0.286567 | 0.2528 | 5.253 | 0.483 | 0.017 |
| 256 | DT | 0.218852 | 0.1939 | 2.581 | 0.092 | 0.013 |
| 256 | reduce | 0.090021 | 0.0607 | 5.664 | 0.127 | 0.021 |

The L1 reduce variant is fastest at every tested tensor extent. At 128 and 256
it is about 2.44x and 2.43x faster than DT, respectively.

The complete 256 run took 26.7 seconds and peaked at 2.38 GiB resident memory.
