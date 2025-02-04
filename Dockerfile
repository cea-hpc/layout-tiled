FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    linux-tools-generic \
    git \
    cmake \
    ninja-build \
    clang \
    libc++-dev \
    libc++abi-dev \
    libomp-dev \
    vim

ENV CC=clang
ENV CXX=clang++

WORKDIR /layout-tiled
COPY . /layout-tiled

RUN cmake \
    -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
    -DLayoutTiled_BUILD_BENCHMARKS=ON \
    -DLayoutTiled_PARALLEL_BENCH=ON

RUN cmake --build build
