# Tiled layouts for `std::mdspan` (C++23)

> [!WARNING]
> This repository is a Work In Progress (WIP).

## Introduction

This project implements tiled layouts for C++23's `std::mdspan`.
Tiled layouts aim to increase the performance of computationally intensive algorithms by improving both spatial and temporal data locality.

## Motivation and scope

As stated in the original MDSPAN paper, [P0009](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html#why-custom-memory-layouts):

> More complex layouts can improve performance significantly for some algorithms. For instance, tiling [...] can improve data locality for many computations relevant to linear algebra and the discretization of partial differential equations. Tiled layouts can also improve vectorization.

When launching a kernel on a GPU, it is mandatory to express the dimensions of blocks of threads ("tiles") that will be scheduled together and are able to leverage a small but fast, shared memory. With `mdspan` standard layouts (`layout_right`, `layout_left`, and `layout_stride`), it is not possible to express tiling as it exists at the hardware level. This leads to more expensive reads from global memory in order to bring the relevant data into the shared memory of a thread block.
Tiled layouts also allow for better and more intuitive cache-blocking on CPU, as the data layout is naturally "cache oblivious".

## Overview & Design decisions

We propose four tiled layouts:

| Layout | Tile order | Element order |
|---|---|---|
| `layout_rtr` | right | right |
| `layout_ltl` | left | left |
| `layout_rtl` | right | left |
| `layout_ltr` | left | right |

The layouts are based on the [Kokkos mdspan tiled layout example](https://github.com/kokkos/mdspan/blob/stable/examples/tiled_layout/simple_tiled_layout.cpp), and extended to support an arbitrary number of dimensions. Tile extents are only supported as compile-time parameters.

## Usage

### Pre-requisites

- C++23 (or later) conforming compiler;
- `cmake` v3.25+;
- `make` or `ninja`.

The library is **header-only** and uses [Kokkos' mdspan reference implementation](https://github.com/kokkos/mdspan), which CMake fetches automatically.

### CMake options reference

| Option | Default | Description |
|---|---|---|
| `LAYOUT_TILED_BUILD_TESTS` | `OFF` | Build the test suite |
| `LAYOUT_TILED_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `LAYOUT_TILED_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `LAYOUT_TILED_ENABLE_DEVICE_OFFLOAD` | `OFF` | Enable CUDA/HIP device offload. |
| `LAYOUT_TILED_BENCH_ENABLE_CHECKS` | `OFF` | Verify benchmark results instead of timing. |
| `LAYOUT_TILED_BENCH_ENABLE_OMP` | `OFF` | Link benchmarks with OpenMP. |
| `LAYOUT_TILED_BENCH_ENABLE_BLAS` | `ON` | Link the GEMM benchmark with BLAS. |

### Build

```bash
cmake -B build -G Ninja

# Optionally build examples / tests / benchmarks:
cmake -B build -G Ninja -DLAYOUT_TILED_BUILD_EXAMPLES=ON -DLAYOUT_TILED_BUILD_TESTS=ON

cmake --build build
```

### Device offload

```bash
cmake -B build -G Ninja -DLAYOUT_TILED_ENABLE_DEVICE_OFFLOAD=ON
```

A working CUDA or HIP compiler is required.

### Install

```bash
cmake -B build
cmake --install build
```

### Run the examples

Four example programs are provided:

```bash
cmake -B build -G Ninja -DLAYOUT_TILED_BUILD_EXAMPLES=ON
cmake --build build

./build/examples/layout_tiled.examples.layout_rtr
./build/examples/layout_tiled.examples.layout_ltl
./build/examples/layout_tiled.examples.layout_rtl
./build/examples/layout_tiled.examples.layout_ltr
```

## Contributing

Contributions are welcome and accepted as [pull requests](https://github.com/dssgabriel/layout-tiled/pulls).
You may also ask questions or file bug reports on the [issue tracker](https://github.com/dssgabriel/layout-tiled/issues).

### Testing

```bash
cmake -B build -G Ninja -DLAYOUT_TILED_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build

# Run a specific test:
ctest --test-dir build -R <TEST_NAME>
```

Tests use GoogleTest (automatically fetched by CMake).

### Benchmarking

```bash
cmake -B build -G Ninja \
  -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_BLAS=OFF
cmake --build build --target layout_tiled.benchmarks.stencil_2d
./build/benchmarks/layout_tiled.benchmarks.stencil_2d
```

Available benchmark names are `gemm`, `stencil_2d`, and `stencil_3d`. Enable OpenMP with `-DLAYOUT_TILED_BENCH_ENABLE_OMP=ON`; the GEMM benchmark can use BLAS when `LAYOUT_TILED_BENCH_ENABLE_BLAS` is enabled.

## License

Copyright (C) 2024-2025 CEA

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or https://www.apache.org/licenses/LICENSE-2.0)

The [SPDX](https://spdx.dev) license identifier for this project is `Apache-2.0`.

Unless required by applicable law or agreed to in writing, software distributed under the Apache-2.0 License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
