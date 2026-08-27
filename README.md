# Tiled layouts for C++23's `std::mdspan`

> [!WARNING]
> This repository is a Work In Progress (WIP).

## Introduction

This project implements tiled layouts for C++23's `std::mdspan`.
Tiled layouts aim to increase the performance of computationally intensive algorithms by improving both spatial and temporal data locality.

## Motivation and scope

As stated in the original MDSPAN paper, [P0009](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html#why-custom-memory-layouts):
> More complex layouts can improve performance significantly for some algorithms. For instance, tiling [...] can improve data locality for many computations relevant to linear algebra and the discretization of partial differential equations. Tiled layouts can also improve vectorization.

When launching a kernel on a GPU, it is mandatory to express the dimensions of blocks of threads ("tiles") that will be scheduled together and are able to leverage a small but fast, shared memory. With `mdspan` standard layouts (`layout_right{_padded}`, `layout_left{_padded}`, and `layout_stride`), it is not possible to express tiling as it exists at the hardware level. This leads to more expensive reads from global memory in order to bring the relevant data into the shared memory of a thread block.
Tiled layouts also allow for better and more intuitive cache-blocking on CPU, as the data layout is naturally "cache oblivious".

## Overview & Design decisions

We propose four tiled layouts:
1. `layout_right_tiled_right`: tiles are stored in *Layout Right* w.r.t each other, and elements are stored in *Layout Right* within tiles;
2. `layout_right_tiled_left`: tiles are stored in *Layout Right* w.r.t each other, and elements are stored in *Layout Left* within tiles;
3. `layout_left_tiled_right`: tiles are stored in *Layout Left* w.r.t each other, and elements are stored in *Layout Right* within tiles;
4. `layout_left_tiled_left`: tiles are stored in *Layout Left* w.r.t each other, and elements are stored in *Layout Left* within tiles.

The proposed layouts are based on [kokkos/mdspan tiled layout example](https://github.com/kokkos/mdspan/blob/stable/examples/tiled_layout/simple_tiled_layout.cpp), and extended to support an arbitrary number of dimensions. Tile extents are only supported as compile-time parameters.


## Usage

### Pre-requisites

Ensure you have the following software installed on your system:
- C++23 conforming compiler;
- `cmake` v3.25+;
- `make` or `ninja`.

This project uses [Kokkos' `mdspan` reference implementation](https://github.com/kokkos/mdspan) (automatically fetched by CMake).

### Build

```bash
cmake -B <BUILDDIR>
cmake --build <BUILDDIR>
```

### Layout-aware reductions

The separate `iai::iai` interface target provides host-only, header-only,
projected N-D execution over the tiled mappings. Axis identifiers are
compile-time `std::size_t` values, and their order in a binding is the dimension
order of that `mdspan`:

```cpp
#include <iai/md_reduce.hpp>

enum : std::size_t { I, J, K };

iai::md_reduce<K>(
  iai::out<I, J>(C, 0.0),
  iai::in<I, K>(A),
  iai::in<K, J>(B),
  [](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
);
```

CMake consumers should link `iai::iai`. The `layout_tiled::layout_tiled`
target contains only the tiled layout mappings.

Use `out<Axes...>(view, identity)` to initialize every register accumulator
from an identity, or `inout<Axes...>(view)` to accumulate from the current
output value. The first argument must be the one output, one or more `in`
bindings follow it, and the updater is last. The updater must mutate its first
argument, accept inputs in binding order, and return `void`.

The union of axis identifiers must be dense from zero, the listed reduction
axes must be unique, and output axes must be exactly the non-reduction axes.
All views that share an axis must have the same logical and static tile extent.
Scalar outputs are not currently supported.

As with the tiled mappings, every logical extent must be divisible by its
physical tile extent. Output and input storage must not overlap. Dynamic shape
disagreements and detectable overlap are checked by assertions in debug
builds; violating either contract in a release build has undefined behavior.

Traversal, cache blocking, the contiguous SIMD axis, reduction-axis order, and
static OpenMP scheduling are inferred from the bindings and tiled mapping
traits. When compiled with OpenMP, output tiles are parallelized; reduction
axes are never split. Arithmetic reductions with contiguous projected inputs
use a portable register-fragment kernel, while all valid calls have a scalar
fallback.

Reduction evaluation is allowed to reassociate according to the selected
layout and backend. Results are therefore not guaranteed to be bitwise
reproducible across builds, targets, or thread counts. This matches the usual
optimized linear-algebra model used by [CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html);
strict reproducibility is likewise a conditional mode in
[oneMKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-2/reproducibility-conditions.html).

The execution strategy, register-fragment design, and a SIMD-friendly GEMV
example are described in [docs/reduce-strategy.md](docs/reduce-strategy.md).
For manual schedule tuning, see
[docs/iai-cache-budgets.md](docs/iai-cache-budgets.md).

### Install

```bash
cmake -B <BUILDDIR>
[sudo] cmake --install <BUILDDIR>
```

### Run the examples

There is one example per layout:
- `examples/layout_right_tiled_right`;
- `examples/layout_right_tiled_left`;
- `examples/layout_left_tiled_right`;
- `examples/layout_left_tiled_left`;

```bash
cmake -B <BUILDDIR> -DLayoutTiled_BUILD_EXAMPLES=ON
cmake --build <BUILDDIR> --target <EXAMPLE>
./<BUILDDIR>/examples/<EXAMPLE>
```

## Contributing

Contributions are welcome and accepted as [pull requests](https://github.com/dssgabriel/layout-tiled/pulls).
You may also ask questions or file bug reports on the [issue tracker](https://github.com/dssgabriel/layout-tiled/issues).

### Testing

**Build & run the tests:**
```bash
cmake -B <BUILDDIR> -DLayoutTiled_BUILD_TESTS=ON
cmake --build <BUILDDIR>
ctest --test-dir <BUILDDIR>
```

### Benchmarking

```bash
cmake -B <BUILDDIR> \
  -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_OMP=ON
cmake --build <BUILDDIR> --target layout_tiled.benchmarks.gemm
```

For controlled GEMM comparisons, an installed BLAS implementation can be
selected without relying on CMake's vendor search:

```bash
cmake -B <BUILDDIR> \
  -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_OMP=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_BLAS=ON \
  -DLAYOUT_TILED_BENCH_BLAS_LIBRARY=/path/to/libopenblasp.so \
  -DLAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR=/path/to/openblas/include

OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES='threads' \
OPENBLAS_NUM_THREADS=4 \
  ./<BUILDDIR>/benchmarks/layout_tiled.benchmarks.gemm \
  --size 1024 --implementation reduce
```

When both OpenMP and BLAS are enabled and no explicit BLAS library is given,
the benchmark configuration prefers an OpenMP-enabled OpenBLAS library and
then a pthread-enabled one before falling back to CMake's generic BLAS search.

The GEMM implementation filter accepts `dt`, `tv`, `reduce`, or `blas`, which
makes it straightforward to collect each sample in a fresh process. When an
OpenBLAS library is linked, BLAS runs report the configured thread count and
reject comparisons that are not using four threads.

For the Phase 1 IAI baseline, first place the CPU in performance mode and,
when possible, lock its policy minimum and maximum to the non-turbo base
frequency. Then run:

```bash
benchmarks/scripts/run_phase1_baseline.sh build/benchmark results/baseline/phase1
```

The runner requires four distinct physical cores, starts each implementation
in a fresh process, and sets both OpenMP and common BLAS thread controls to
four. Its default placement is quoted as `OMP_PLACES='{0},{1},{2},{3}'` and the
process is confined with `taskset --cpu-list 0,1,2,3`. These numbers are Linux
CPU IDs (hwloc `P#` indices), not the logical `L#` indices printed by hwloc.
The runner rejects SMT-sibling selections and, by default, any CPU policy whose
minimum and maximum are not both equal to its base frequency.

Raw CSVs, per-process stdout/stderr, verbose timing information, topology, and
frequency metadata are written below the requested result directory.

## License

Copyright (C) 2024-2025 CEA

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or https://www.apache.org/licenses/LICENSE-2.0)

The [SPDX](https://spdx.dev) license identifier for this project is `Apache-2.0`.

Unless required by applicable law or agreed to in writing, software distributed under the Apache-2.0 License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
