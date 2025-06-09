# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
linux-tools-generic \
build-essential \
ca-certificates \
git \
cmake \
ninja-build \
clang \
libomp-dev \
libopenblas-dev \
python3 \
numactl \
vim

WORKDIR /layout-tiled
COPY . /layout-tiled

RUN cmake \
-B build \
-G 'Ninja Multi-Config' \
-DLayoutTiled_BUILD_BENCHMARKS=ON \
-DLayoutTiled_BENCHMARKS_ENABLE_OPENMP=ON \
-DLayoutTiled_BENCHMARKS_ENABLE_BLAS=ON

RUN cmake --build build --config Release
