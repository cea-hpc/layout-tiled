FROM ubuntu:24.04 AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        linux-tools-generic \
        build-essential \
        ca-certificates \
        git \
        cmake \
        ninja-build \
        clang \
        libomp-dev \
        libopenblas-dev

WORKDIR /layout-tiled

COPY include/ ./include
COPY tests/ ./tests
COPY benchmarks/ ./benchmarks
COPY examples/ ./examples
COPY cmake/ ./cmake
COPY CMakeLists.txt .

WORKDIR /layout-tiled/build-gcc

RUN cmake \
        -G 'Ninja Multi-Config' \
        -B . \
        -S .. \
        -DCMAKE_CXX_COMPILER=g++ \
        -DLayoutTiled_BUILD_BENCHMARKS=ON \
        -DLayoutTiled_BENCHMARKS_ENABLE_OPENMP=ON \
        -DLayoutTiled_BENCHMARKS_ENABLE_BLAS=ON && \
    cmake --build . --config Release

WORKDIR /layout-tiled/build-clang

RUN cmake \
        -G 'Ninja Multi-Config' \
        -B . \
        -S .. \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLayoutTiled_BUILD_BENCHMARKS=ON \
        -DLayoutTiled_BENCHMARKS_ENABLE_OPENMP=ON \
        -DLayoutTiled_BENCHMARKS_ENABLE_BLAS=ON && \
    cmake --build . --config Release

FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        linux-tools-generic \
        build-essential \
        ca-certificates \
        git \
        libomp \
        libopenblas \
        python3 \
        vim

RUN addgroup -S lt && adduser -S lt -G lt
USER lt

COPY --chown=lt:lt --from=build \
    ./layout-tiled/build-gcc
    ./gcc

COPY --chown=lt:lt --from=build \
    ./layout-tiled/build-gcc \
    ./gcc

COPY --chown=lt:lt --from=build \
    ./layout-tiled/build-clang
    ./clang

COPY --chown=lt:lt --from=build \
    ./layout-tiled/benchmarks/scripts
    ./scripts
