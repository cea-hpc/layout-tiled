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

#include <cstddef>
#include <iostream>
#include <mdspan>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nanobench.h>
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#include <omp.h>
#endif

#include <layout_tiled/layout_right_tiled_right.hpp>

using namespace std::chrono_literals;

template <class T, class Extents, class Layout, class AccessorPolicy>
auto ijk(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <class T, class Extents, class Layout, class AccessorPolicy>
auto ikj(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T A_ik = A[i, k];
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <class T, class Extents, class Layout, class AccessorPolicy>
auto jik(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T acc{};
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <class T, class Extents, class Layout, class AccessorPolicy>
auto jki(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T B_kj = B[k, j];
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

template <class T, class Extents, class Layout, class AccessorPolicy>
auto kij(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T A_ik = A[i, k];
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <class T, class Extents, class Layout, class AccessorPolicy>
auto kji(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T B_kj = B[k, j];
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp simd
#endif
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto ijk_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//     auto const i_end = std::min(bi + TILE, A.extent(0));
//     for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//       auto const j_end = std::min(bj + TILE, B.extent(1));
//       for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//         auto const k_end = std::min(bk + TILE, A.extent(1));

//         for (std::size_t i = bi; i < i_end; ++i) {
//           for (std::size_t j = bj; j < j_end; ++j) {
//             T acc{};
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t k = bk; k < k_end; ++k) {
//               acc += A[i, k] * B[k, j];
//             }
//             C[i, j] = acc;
//           }
//         }
//       }
//     }
//   }
// }

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto ikj_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//     auto const i_end = std::min(bi + TILE, A.extent(0));
//     for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//       auto const k_end = std::min(bk + TILE, A.extent(1));
//       for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//         auto const j_end = std::min(bj + TILE, B.extent(1));

//         for (std::size_t i = bi; i < i_end; ++i) {
//           for (std::size_t k = bk; k < k_end; ++k) {
//             T A_ik = A[i, k];
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t j = bj; j < j_end; ++j) {
//               C[i, j] += A_ik * B[k, j];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto jik_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//     auto const j_end = std::min(bj + TILE, B.extent(1));
//     for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//       auto const i_end = std::min(bi + TILE, A.extent(0));
//       for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//         auto const k_end = std::min(bk + TILE, A.extent(1));

//         for (std::size_t j = bj; j < j_end; ++j) {
//           for (std::size_t i = bi; i < i_end; ++i) {
//             T acc{};
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t k = bk; k < k_end; ++k) {
//               acc += A[i, k] * B[k, j];
//             }
//             C[i, j] = acc;
//           }
//         }
//       }
//     }
//   }
// }

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto jki_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//     auto const j_end = std::min(bj + TILE, B.extent(1));
//     for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//       auto const k_end = std::min(bk + TILE, A.extent(1));
//       for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//         auto const i_end = std::min(bi + TILE, A.extent(0));

//         for (std::size_t j = bj; j < j_end; ++j) {
//           for (std::size_t k = bk; k < k_end; ++k) {
//             T B_kj = B[k, j];
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t i = bi; i < i_end; ++i) {
//               C[i, j] += A[i, k] * B_kj;
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto kij_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//     auto const k_end = std::min(bk + TILE, A.extent(1));
//     for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//       auto const i_end = std::min(bi + TILE, A.extent(0));
//       for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//         auto const j_end = std::min(bj + TILE, B.extent(1));

//         for (std::size_t k = bk; k < k_end; ++k) {
//           for (std::size_t i = bi; i < i_end; ++i) {
//             T A_ik = A[i, k];
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t j = bj; j < j_end; ++j) {
//               C[i, j] += A_ik * B[k, j];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <class T, class Extents, class Layout, class AccessorPolicy>
// auto kji_block(
//   std::mdspan<T, Extents, Layout, AccessorPolicy> A,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> B,
//   std::mdspan<T, Extents, Layout, AccessorPolicy> C
// ) -> void {
//   assert(A.extent(1) == B.extent(0));
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));

// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//   #pragma omp parallel for collapse(2) schedule(static)
// #endif
//   for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
//     auto const k_end = std::min(bk + TILE, A.extent(1));
//     for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
//       auto const j_end = std::min(bj + TILE, B.extent(1));
//       for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
//         auto const i_end = std::min(bi + TILE, A.extent(0));

//         for (std::size_t k = bk; k < k_end; ++k) {
//           for (std::size_t j = bj; j < j_end; ++j) {
//             T B_kj = B[k, j];
// #if defined(LAYOUT_TILED_PARALLEL_BENCH)
//             #pragma omp simd
// #endif
//             for (std::size_t i = bi; i < i_end; ++i) {
//               C[i, j] += A[i, k] * B_kj;
//             }
//           }
//         }
//       }
//     }
//   }
// }

auto main() -> int {
  using extents_type = std::dextents<std::size_t, 2>;

  constexpr std::pair<size_t, size_t> DIMS[] = {
    {512zu, 512zu},
    {768zu, 768zu},
    {1024zu, 1024zu},
    {1280zu, 1280zu},
    {1536zu, 1536zu},
    // {1792zu, 1792zu},
    {2048zu, 2048zu},
    // {2304zu, 2304zu},
    {2560zu, 2560zu},
    // {2816zu, 2816zu},
    {3072zu, 3072zu},
    // {3328zu, 3328zu},
    {3584zu, 3584zu},
    // {3840zu, 3840zu},
    {4096zu, 4096zu},
  };

  ankerl::nanobench::Bench bs;
  bs.output(nullptr);

  for (auto const& dims : DIMS) {
    // Standard layout matrices
    std::vector<double> A(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> B(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> C(std::get<0>(dims) * std::get<1>(dims));
    auto mA = std::mdspan(A.data(), std::get<0>(dims), std::get<1>(dims));
    auto mB = std::mdspan(B.data(), std::get<0>(dims), std::get<1>(dims));
    auto mC = std::mdspan(C.data(), std::get<0>(dims), std::get<1>(dims));

    // Tiled layout matrices
    std::vector<double> tA_32(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tB_32(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tC_32(std::get<0>(dims) * std::get<1>(dims));
    auto mA_tiled_32 = std::mdspan(
      tA_32.data(),
      layout_right_tiled_right<32, 32>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mB_tiled_32 = std::mdspan(
      tB_32.data(),
      layout_right_tiled_right<32, 32>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mC_tiled_32 = std::mdspan(
      tC_32.data(),
      layout_right_tiled_right<32, 32>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );

    std::vector<double> tA_64(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tB_64(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tC_64(std::get<0>(dims) * std::get<1>(dims));
    auto mA_tiled_64 = std::mdspan(
      tA_64.data(),
      layout_right_tiled_right<64, 64>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mB_tiled_64 = std::mdspan(
      tB_64.data(),
      layout_right_tiled_right<64, 64>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mC_tiled_64 = std::mdspan(
      tC_64.data(),
      layout_right_tiled_right<64, 64>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );

    std::vector<double> tA_128(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tB_128(std::get<0>(dims) * std::get<1>(dims));
    std::vector<double> tC_128(std::get<0>(dims) * std::get<1>(dims));
    auto mA_tiled_128 = std::mdspan(
      tA_128.data(),
      layout_right_tiled_right<128, 128>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mB_tiled_128 = std::mdspan(
      tB_128.data(),
      layout_right_tiled_right<128, 128>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );
    auto mC_tiled_128 = std::mdspan(
      tC_128.data(),
      layout_right_tiled_right<128, 128>::mapping<extents_type>(extents_type(std::get<0>(dims), std::get<1>(dims)))
    );

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::size_t i = 0; i < std::get<0>(dims); ++i) {
      for (std::size_t j = 0; j < std::get<1>(dims); ++j) {
        mA[i, j] = 1.0;
        mB[i, j] = 1.0;
        mC[i, j] = 1.0;

        mA_tiled_32[i, j]  = 1.0;
        mB_tiled_32[i, j]  = 1.0;
        mC_tiled_32[i, j]  = 1.0;
        mA_tiled_64[i, j]  = 1.0;
        mB_tiled_64[i, j]  = 1.0;
        mC_tiled_64[i, j]  = 1.0;
        mA_tiled_128[i, j] = 1.0;
        mB_tiled_128[i, j] = 1.0;
        mC_tiled_128[i, j] = 1.0;
      }
    }

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
    bs.title(fmt::format("{}C {}x{}", omp_get_max_threads(), std::get<0>(dims), std::get<1>(dims)))
      .timeUnit(1s, "s")
      .relative(true);

    bs.run("IJK standard layout", [&]() { ijk(mA, mB, mC); });
    bs.run("IJK tiled layout 32x32", [&]() { ijk(mA_tiled_32, mB_tiled_32, mC_tiled_32); });
    bs.run("IJK tiled layout 64x64", [&]() { ijk(mA_tiled_64, mB_tiled_64, mC_tiled_64); });
    bs.run("IJK tiled layout 128x128", [&]() { ijk(mA_tiled_128, mB_tiled_128, mC_tiled_128); });
    // bs.run("ijk block standard", [&]() { ijk_block(mA, mB, mC); });
    // bs.run("ijk block tiled", [&]() { ijk_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("ikj standard", [&]() { ikj(mA, mB, mC); });
    // bs.run("ikj tiled", [&]() { ikj(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("ikj block standard", [&]() { ikj_block(mA, mB, mC); });
    // bs.run("ikj block tiled", [&]() { ikj_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("jik standard", [&]() { jik(mA, mB, mC); });
    // bs.run("jik tiled", [&]() { jik(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("jik block standard", [&]() { jik_block(mA, mB, mC); });
    // bs.run("jik block tiled", [&]() { jik_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("jki standard", [&]() { jki(mA, mB, mC); });
    // bs.run("jki tiled", [&]() { jki(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("jki block standard", [&]() { jki_block(mA, mB, mC); });
    // bs.run("jki block tiled", [&]() { jki_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("kij standard", [&]() { kij(mA, mB, mC); });
    // bs.run("kij tiled", [&]() { kij(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("kij block standard", [&]() { kij_block(mA, mB, mC); });
    // bs.run("kij block tiled", [&]() { kij_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("kji standard", [&]() { kji(mA, mB, mC); });
    // bs.run("kji tiled", [&]() { kji(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("kji block standard", [&]() { kji_block(mA, mB, mC); });
    // bs.run("kji block tiled", [&]() { kji_block(mA_tiled, mB_tiled, mC_tiled); });
#else
    bs.title(fmt::format("serial {}x{}", std::get<0>(dims), std::get<1>(dims))).timeUnit(1s, "s").relative(true);

    bs.run("IJK standard layout", [&]() { ijk(mA, mB, mC); });
    bs.run("IJK tiled layout 32x32", [&]() { ijk(mA_tiled_32, mB_tiled_32, mC_tiled_32); });
    bs.run("IJK tiled layout 64x64", [&]() { ijk(mA_tiled_64, mB_tiled_64, mC_tiled_64); });
    bs.run("IJK tiled layout 128x128", [&]() { ijk(mA_tiled_128, mB_tiled_128, mC_tiled_128); });
    // bs.run("ijk block standard", [&]() { ijk_block(mA, mB, mC); });
    // bs.run("ijk block tiled", [&]() { ijk_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("ikj standard", [&]() { ikj(mA, mB, mC); });
    // bs.run("ikj tiled", [&]() { ikj(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("ikj block standard", [&]() { ikj_block(mA, mB, mC); });
    // bs.run("ikj block tiled", [&]() { ikj_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("jik standard", [&]() { jik(mA, mB, mC); });
    // bs.run("jik tiled", [&]() { jik(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("jik block standard", [&]() { jik_block(mA, mB, mC); });
    // bs.run("jik block tiled", [&]() { jik_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("jki standard", [&]() { jki(mA, mB, mC); });
    // bs.run("jki tiled", [&]() { jki(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("jki block standard", [&]() { jki_block(mA, mB, mC); });
    // bs.run("jki block tiled", [&]() { jki_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("kij standard", [&]() { kij(mA, mB, mC); });
    // bs.run("kij tiled", [&]() { kij(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("kij block standard", [&]() { kij_block(mA, mB, mC); });
    // bs.run("kij block tiled", [&]() { kij_block(mA_tiled, mB_tiled, mC_tiled); });

    // bs.run("kji standard", [&]() { kji(mA, mB, mC); });
    // bs.run("kji tiled", [&]() { kji(mA_tiled, mB_tiled, mC_tiled); });
    // bs.run("kji block standard", [&]() { kji_block(mA, mB, mC); });
    // bs.run("kji block tiled", [&]() { kji_block(mA_tiled, mB_tiled, mC_tiled); });
#endif
    bs.render(ankerl::nanobench::templates::csv(), std::cout);
  }

  return 0;
}
