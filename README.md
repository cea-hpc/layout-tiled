# Tiled layouts for C++23's `std::mdspan`

> [!WARNING]
> This repository is a Work In Progress (WIP).


## Motivation

This project implements tiled layouts for C++23's `std::mdspan`.
Tiled layouts aim to increase the performance of computationally intensive algorithms by improving both spatial and temporal data locality.   
As stated in the original MDSPAN paper, [P0009](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html#why-custom-memory-layouts):
> More complex layouts can improve performance significantly for some algorithms. For instance, tiling [...] can improve data locality for many computations relevant to linear algebra and the discretization of partial differential equations. Tiled layouts can also improve vectorization.

When launching a kernel on a GPU, it is mandatory to express the dimensions of blocks of threads ("tiles") that will be scheduled together and are able to leverage a small but fast, shared memory. With `mdspan` standard layouts (`layout_right{_padded}`, `layout_left{_padded}`, and `layout_stride`), it is not possible to express tiling as it exists at the hardware level. This leads to more expensive reads from global memory in order to bring the relevant data into the shared memory of a thread block.   
Tiled layouts also allow for better and more intuitive cache-blocking on CPU, as the data layout is naturally "cache oblivious".


## Overview

We propose four tiled layouts (based on Z-order curves):
1. `layout_right_tiled_right`: tiles are stored in *Layout Right* w.r.t each other, and elements are stored in *Layout Right* within tiles;
2. `layout_right_tiled_left`: tiles are stored in *Layout Right* w.r.t each other, and elements are stored in *Layout Left* within tiles;
3. `layout_left_tiled_right`: tiles are stored in *Layout Left* w.r.t each other, and elements are stored in *Layout Right* within tiles;
4. `layout_left_tiled_left`: tiles are stored in *Layout Left* w.r.t each other, and elements are stored in *Layout Left* within tiles.

The proposed layouts are currently based on [kokkos/mdspan tiled layout example](https://github.com/kokkos/mdspan/blob/stable/examples/tiled_layout/simple_tiled_layout.cpp). They are limited to 2-dimensional `mdspan`s, with tiles of the same rank. Tile extents are only supported as runtime parameters.

### Goals
- Support both static _and_ runtime extents for tiles;
- Support `mdspan`s of arbitrary rank.

### Open Design Questions
- Do we allow for tiles to have a different rank (necessarily smaller) than the `mdspan` they map? (e.g. given a 3-dimensional span, can we make tiles that are only 2-dimensional?)


## Usage

### Pre-requisites

Ensure you have the following software installed on your system:
- C++23 conforming compiler;
- C++ standard libray with `std::mdspan` support (e.g., LLVM's `libcxx` v17+);
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
./<BUILDDIR>/examples/<EXAMPLE>
```
