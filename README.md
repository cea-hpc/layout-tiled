# Tiled layouts for C++23's `std::mdspan`

## Usage

### Pre-requisites

- C++23 conforming compiler with `std::mdspan` support (e.g., `clang` v17.0+);
- `cmake` v3.23+;
- `make` or `ninja`.

### Build

```bash
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++-17
cmake --build build
```

### Run examples

```bash
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++-17 -Dlayout_tiled_BUILD_EXAMPLES=ON
cmake --build build --target simple
./build/simple
```
