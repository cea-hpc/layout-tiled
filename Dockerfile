FROM alpine:3.22 AS build

RUN apk update && \
    apk add --no-cache \
        linux-headers=6.14.2-r0 \
        build-base=0.5-r3 \
        git=2.49.1-r0 \
        cmake=3.31.7-r1 \
        gcc=14.2.0-r6 \
        clang18=18.1.8-r5 \
        openmp-dev=18.1.8-r2 \
        openblas-dev=0.3.28-r0

WORKDIR /layout-tiled

COPY include/ ./include
COPY benchmarks/ ./benchmarks
COPY cmake/ ./cmake
COPY CMakeLists.txt .

WORKDIR /layout-tiled/build

RUN cmake \
        -B gcc \
        -S .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
        -DLAYOUT_TILED_BENCH_ENABLE_OMP=ON \
        -DLAYOUT_TILED_BENCH_ENABLE_BLAS=ON && \
    cmake --build gcc -j

RUN cmake \
        -B llvm \
        -S .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=clang++-18 \
        -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
        -DLAYOUT_TILED_BENCH_ENABLE_OMP=ON \
        -DLAYOUT_TILED_BENCH_ENABLE_BLAS=ON && \
    cmake --build llvm -j

FROM alpine:3.22

RUN apk update && \
    apk add --no-cache \
        build-base=0.5-r3 \
        openmp=18.1.8-r2 \
        openblas=0.3.28-r0 \
        python3=3.12.11-r0 \
        vim=9.1.1566-r0

RUN addgroup -S lt && adduser -S lt -G lt
USER lt

COPY --chown=lt:lt --from=build \
    ./layout-tiled/build/gcc \
    ./layout-tiled/build/gcc

COPY --chown=lt:lt --from=build \
    ./layout-tiled/build/llvm \
    ./layout-tiled/build/llvm

COPY --chown=lt:lt --from=build \
    ./layout-tiled/benchmarks/scripts \
    ./layout-tiled/scripts
