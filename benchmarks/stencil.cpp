/// Copyright (C) 2024, CEA
///
/// Licensed under the MIT License ("the License" hereafter);
/// You may not use this file except in compliance with the License.
/// You should have received a full copy of the License along with this program;
/// if not, you can obtain a copy at:
///
///    https://opensource.org/license/MIT
///
/// The software is provided “as is”, without warranty of any kind, express
/// or implied, including but not limited to the warranties of merchantability,
/// fitness for a particular purpose and noninfringement. In no event shall
/// the authors or copyright holders be liable for any claim, damages or other
/// liability, whether in an action of contract, tort or otherwise, arising from,
/// out of or in connection with the software or the use or other dealings
/// in the software.
///
/// Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <mdspan>
#include <string>
#include <utility>
#include <vector>

#include <getopt.h>

#include <fmt/format.h>
#include <nanobench.h>
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#include <omp.h>
#endif

#include <layout_tiled/layout_right_tiled_right.hpp>

using namespace std::chrono_literals;

static constexpr size_t ORDER      = 16;
static constexpr size_t HALF_ORDER = ORDER / 2;
static constexpr double EXPONENT   = 17.0;

// Elevate double-precision floating-point value to power `n`, similar to Rust's `f64::powi`
static inline constexpr auto powi_f64(double value, int n) -> double {
  if (n == 0) {
    return 1.0;
  }
  double result = 1.0;
  for (int _ = 0; _ < n; ++_) {
    result *= value;
  }
  return n >= 0 ? result : 1.0 / result;
};

template <class T, class Extents, class Layout, class AccessorPolicy>
static auto stencil(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C,
  std::array<double, HALF_ORDER> exponents
) -> void {
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t i = HALF_ORDER; i < A.extent(0) - HALF_ORDER; ++i) {
    for (size_t j = HALF_ORDER; j < A.extent(1) - HALF_ORDER; ++j) {
      for (size_t k = HALF_ORDER; k < A.extent(2) - HALF_ORDER; ++k) {
        A[i, j, k] *= B[i, j, k];
      }
    }
  }

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t i = HALF_ORDER; i < C.extent(0) - HALF_ORDER; ++i) {
    for (size_t j = HALF_ORDER; j < C.extent(1) - HALF_ORDER; ++j) {
      for (size_t k = HALF_ORDER; k < C.extent(2) - HALF_ORDER; ++k) {
        double acc = A[i, j, k];
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd reduction(+ : acc)
#endif
        for (size_t n = 1; n < HALF_ORDER + 1; ++n) {
          acc += A[i - n, j, k] * exponents[n - 1];
          acc += A[i + n, j, k] * exponents[n - 1];
          acc += A[i, j - n, k] * exponents[n - 1];
          acc += A[i, j + n, k] * exponents[n - 1];
          acc += A[i, j, k - n] * exponents[n - 1];
          acc += A[i, j, k + n] * exponents[n - 1];
        }
        C[i, j, k] = acc;
      }
    }
  }
  std::swap(A, C);
}

auto main(int32_t argc, char* argv[]) -> int32_t {
  // static std::string HELP = "Usage: layout_tiled.benchmarks.gemm [OPTIONS]\n\n"
  //                           "Options:\n"
  //                           "  -s, --strong-scaling   Run strong scaling\n"
  //                           "  -w, --weak-scaling     Run weak scaling\n"
  //                           "  -c, --perf-comp        Run performance comparison\n"
  //                           "  -h, --help             Show this help\n";

  using extents_type = std::dextents<size_t, 3>;

  // Exponents array initialization
  constexpr auto exponents = []() {
    std::array<double, HALF_ORDER> tmp{};
    for (size_t n = 0; n < HALF_ORDER; ++n) {
      tmp[n] = 1.0 / powi_f64(EXPONENT, int(n + 1));
    }
    return tmp;
  }();

  ankerl::nanobench::Bench bs;
  bs.output(nullptr);

  constexpr std::tuple<size_t, size_t, size_t> DIMS[] = {
    {128 - ORDER, 128 - ORDER, 128 - ORDER},
    {256 - ORDER, 256 - ORDER, 256 - ORDER},
    // {512 - ORDER, 512 - ORDER, 512 - ORDER},
    // {1024 - ORDER, 1024 - ORDER, 1024 - ORDER},
  };
  for (auto dims : DIMS) {
    size_t dimx = std::get<0>(dims);
    size_t dimy = std::get<1>(dims);
    size_t dimz = std::get<2>(dims);
    size_t maxx = dimx + ORDER;
    size_t maxy = dimy + ORDER;
    size_t maxz = dimz + ORDER;

    std::vector<double> A(maxx * maxy * maxz);
    std::vector<double> B(maxx * maxy * maxz);
    std::vector<double> C(maxx * maxy * maxz);
    auto mA = std::mdspan(A.data(), maxx, maxy, maxz);
    auto mB = std::mdspan(B.data(), maxx, maxy, maxz);
    auto mC = std::mdspan(C.data(), maxx, maxy, maxz);

    std::vector<double> tA(maxx * maxy * maxz);
    std::vector<double> tB(maxx * maxy * maxz);
    std::vector<double> tC(maxx * maxy * maxz);
    auto mA_tiled = std::mdspan(
      tA.data(), layout_right_tiled_right<32, 32, 32>::mapping<extents_type>(extents_type(maxx, maxy, maxz))
    );
    auto mB_tiled = std::mdspan(
      tB.data(), layout_right_tiled_right<32, 32, 32>::mapping<extents_type>(extents_type(maxx, maxy, maxz))
    );
    auto mC_tiled = std::mdspan(
      tC.data(), layout_right_tiled_right<32, 32, 32>::mapping<extents_type>(extents_type(maxx, maxy, maxz))
    );

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(3) schedule(static)
#endif
    // Tensors initialization
    for (size_t i = 0; i < maxx; ++i) {
      for (size_t j = 0; j < maxy; ++j) {
        for (size_t k = 0; k < maxz; ++k) {
          if (i >= HALF_ORDER && i < dimx + HALF_ORDER
              && j >= HALF_ORDER && j < dimy + HALF_ORDER
              && k >= HALF_ORDER && k < dimz + HALF_ORDER
          ) {
            mA[i, j, k]       = 1.0;
            mA_tiled[i, j, k] = 1.0;
          } else {
            mA[i, j, k]       = 0.0;
            mA_tiled[i, j, k] = 0.0;
          }
          mB[i, j, k] = sin(
            static_cast<double>(k) * cos(static_cast<double>(j) + 0.817) * cos(static_cast<double>(i) + 0.311) + 0.613
          );
          mB_tiled[i, j, k] = sin(
            static_cast<double>(k) * cos(static_cast<double>(j) + 0.817) * cos(static_cast<double>(i) + 0.311) + 0.613
          );
          mC[i, j, k]       = 0.0;
          mC_tiled[i, j, k] = 0.0;
        }
      }
    }

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
    bs.title(fmt::format("{}C {}x{}x{}", omp_get_max_threads(), dimx, dimy, dimz)).timeUnit(1s, "s").relative(true);
#else
    bs.title(fmt::format("{}x{}x{}", dimx, dimy, dimz)).timeUnit(1s, "s").relative(true);
#endif

    bs.run("stencil standard layout", [&]() { stencil(mA, mB, mC, exponents); });
    bs.run("stencil tiled layout 32x32x32", [&]() { stencil(mA_tiled, mB_tiled, mC_tiled, exponents); });

    bs.render(ankerl::nanobench::templates::csv(), std::cout);
  }

  return 0;
}
