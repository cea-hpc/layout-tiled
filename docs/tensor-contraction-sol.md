# Tensor contraction performance relative to the P-core Speed of Light

These measurements characterize the FP64 tensor contraction

```text
C[b,i,j] = sum_k A[b,i,k] * B[b,k,j]
```

on the Intel Core Ultra 5 135H laptop used for development. They were recorded
on 2026-08-02 with a problem extent of 128 in every dimension. Each reported
time is the median of five fresh-process runs.

## P-core compute ceiling

The processor has four Redwood Cove performance cores. Intel specifies a
1.7 GHz P-core base frequency, a maximum 4.6 GHz single-core turbo frequency,
AVX2, and an 18 MB shared cache. See the
[Intel Core Ultra 5 135H specifications](https://www.intel.com/content/www/us/en/products/sku/236850/intel-core-ultra-5-processor-135h-18m-cache-up-to-4-60-ghz/specifications.html).
Intel's throughput package states that Redwood Cove has the same instruction
throughput as Golden Cove; see
[Intel's Redwood Cove throughput documentation](https://www.intel.com/content/www/us/en/content-details/825952/intel-processors-and-processor-cores-based-on-crestmont-and-redwood-cove-microarchitecture-instruction-throughput-and-latency.html).

A 256-bit AVX2 vector holds four doubles. Two vector FMAs per cycle, counting
the multiply and add separately, give

```text
4 doubles/vector × 2 FLOP/FMA × 2 vector FMA/cycle
  = 16 FP64 FLOP/cycle/P-core.
```

The useful ceilings for this system are:

| Ceiling | Calculation | FP64 GFLOP/s |
|---|---:|---:|
| Base-clock reference | 4 × 1.7 GHz × 16 | 108.8 |
| Frequency-policy ceiling observed during testing | 4 × 2.237 GHz × 16 | 143.2 |
| Sustained register-only AVX2 FMA microkernel, five-run median | measured | 131.6 |
| Absolute single-core-turbo extrapolation | 4 × 4.6 GHz × 16 | 294.4 |

The 294.4 GFLOP/s extrapolation is not a realistic all-core laptop target.
Intel defines 4.6 GHz as a maximum single-core turbo frequency. The sustained
131.6 GFLOP/s register-only result is therefore the practical P-core Speed of
Light used below. Assembly inspection confirmed twelve independent YMM FMA
accumulator chains with no data loads or stores in the timed reduction loop.

## Benchmark placement and operation count

The benchmark used one hardware thread on each of the four P-cores:

```sh
OMP_NUM_THREADS=4 \
OMP_PROC_BIND=spread \
OMP_PLACES='{0},{2},{4},{6}' \
OPENBLAS_NUM_THREADS=4 \
./build/benchmarks/Release/layout_tiled.benchmarks.tensor_contraction \
  --size 128
```

This explicit placement matters. The earlier generic
`OMP_PLACES=threads OMP_PROC_BIND=spread` configuration selected two P-cores
and two E-cores on this hybrid processor and cannot be compared directly with
a four-P-core ceiling.

There are `128^3` output elements, each with a length-128 reduction. Counting
an FMA as two floating-point operations gives

```text
2 × 128^4 = 536,870,912 FLOP.
```

## Results

The table reports the fastest tile configuration for each implementation from
the rectangular-tile sweep.

| Implementation | Physical tile `(B,M,N,K)` | Median time | GFLOP/s | Practical P-core SoL |
|---|---:|---:|---:|---:|
| `reduce` | `(1,32,64,32)` | 5.601 ms | 95.85 | 72.8% |
| Data tiled (DT) | `(4,16,64,32)` | 10.783 ms | 49.79 | 37.8% |
| Cache blocked (CB) | `(1,32,64,32)` | 11.611 ms | 46.24 | 35.1% |

The `reduce` implementation is 1.93 times faster than the best DT case and
2.07 times faster than the best cache-blocked case. Relative to the
frequency-policy-derived 143.2 GFLOP/s ceiling it reaches 66.9%; relative to
the register-only sustained ceiling it reaches 72.8%.

## Interpretation

The contraction has an ideal arithmetic intensity of approximately
10.67 FLOP/byte if each input is fetched once and the output is written once,
or approximately 8 FLOP/byte when conservatively including output write
allocation. Reaching the measured compute roof would consequently require
about 12.3--16.5 GB/s after effective cache reuse. The remaining gap is more
consistent with real-kernel instruction overhead, cache traffic, address
generation, and incomplete FMA-pipe occupancy than with the nominal DRAM
bandwidth limit.

Holding each 4-by-8 output fragment in registers across the complete reduction
is what moves `reduce` close to this ceiling. The tiled layouts make the
projected input accesses contiguous and keep the input working set reusable;
the axis description lets the executor select that fragment schedule without
encoding tensor-contraction-specific loops in the public API.
