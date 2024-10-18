# Tiled layouts for C++23's `std::mdspan`

> [!WARNING]
> This repository is a Work In Progress (WIP).


## Motivation

This project implements tiled layouts for C++23's `std::mdspan`.
Tiled layouts aim to increase the performance of computationally intensive algorithms by improving both spatial and temporal data locality.
As stated in the original MDSPAN paper, [P0009](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html#why-custom-memory-layouts):
> More complex layouts can improve performance significantly for some algorithms. For instance, tiling [...] can improve data locality for many computations relevant to linear algebra and the discretization of partial differential equations. Tiled layouts can also improve vectorization.

When launching a kernel on a GPU, it is mandatory to express the dimensions of blocks of threads ("tiles") that will be scheduled together and are able to leverage a small but fast, shared memory. With `mdspan` standard layouts (`layout_right{_padded}`, `layout_left{_padded}`, and `layout_stride`), it is not possible to express tiling, as it exists at the hardware level. This leads to more expensive reads from global memory in order to bring the relevant data into the shared memory of a thread block. Using tiled data layouts may significantly improve the performance of some algorithms, e.g. matrix multiplication (GEMM).


## Overview

We propose four tiled layouts:
1. `layout_right_tiled_right`: elements are stored in *Layout Right* within tiles, and tiles are stored in *Layout Right* w.r.t each other;
2. `layout_right_tiled_left`: elements are stored in *Layout Left* within tiles, and tiles are stored in *Layout Right* w.r.t each other;
3. `layout_left_tiled_right`: elements are stored in *Layout Right* within tiles, and tiles are stored in *Layout Left* w.r.t each other;
4. `layout_left_tiled_left`: elements are stored in *Layout Left* within tiles, and tiles are stored in *Layout Left* w.r.t each other;

The proposed layouts are currently limited to 2-dimensional `mdspan`s, with tiles of the same rank. Tile dimensions are only supported as runtime parameters.

### Goals
- Support both static _and_ runtime extents for tiles;
- Support `mdspan`s of arbitrary rank.

### Open Design Questions
- Do we allow for tiles to not have the same rank as the `mdspan`? (e.g. given a 3-dimensional span, can we make tiles that are only 2-dimensional?)


## Usage

### Pre-requisites

- C++23 conforming compiler;
- C++ standard libray with `std::mdspan` support (e.g., LLVM's `libcxx` v17.0+);
- `cmake` v3.23+;
- `make` or `ninja`.

### Build

```bash
cmake -B <BUILDDIR>
cmake --build <BUILDDIR>
```

### Install

```bash
cmake -B <BUILDDIR>
cmake --build <BUILDDIR> --target install
```

### Run the examples

There is one example per layout:
- `examples/layout_right_tiled_right`;
- `examples/layout_right_tiled_left`;
- `examples/layout_left_tiled_right`;
- `examples/layout_left_tiled_left`;

```bash
cmake -B <BUILDDIR> -Dlayout_tiled_BUILD_EXAMPLES=ON
cmake --build <BUILDDIR> --target <EXAMPLE>
./<BUILDDIR>/<EXAMPLE>
```
