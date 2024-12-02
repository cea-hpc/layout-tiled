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

#include <layout_tiled/layout_left_tiled_left.hpp>
#include <layout_tiled/layout_left_tiled_right.hpp>
#include <layout_tiled/layout_right_tiled_left.hpp>
#include <layout_tiled/layout_right_tiled_right.hpp>

#include <nanobench.h>

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#include <omp.h>
#endif

#include <array>
#include <format>
#include <mdspan>
#include <vector>

using namespace std::chrono_literals;

constexpr std::size_t TILE = 32;

namespace serial {

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ijk(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ikj(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T A_ik = A[i, k];
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jik(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T acc{};
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jki(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T B_kj = B[k, j];
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kij(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T A_ik = A[i, k];
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kji(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T B_kj = B[k, j];
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ijk_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
    auto const i_end = std::min(bi + TILE, A.extent(0));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
      auto const j_end = std::min(bj + TILE, B.extent(1));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
        auto const k_end = std::min(bk + TILE, A.extent(1));

        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T acc{};
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ikj_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
    auto const i_end = std::min(bi + TILE, A.extent(0));
    for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
      auto const k_end = std::min(bk + TILE, A.extent(1));
      for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
        auto const j_end = std::min(bj + TILE, B.extent(1));

        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t k = bk; k < k_end; ++k) {
            T A_ik = A[i, k];
            for (std::size_t j = bj; j < j_end; ++j) {
              C[i, j] += A_ik * B[k, j];
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jik_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
    auto const j_end = std::min(bj + TILE, B.extent(1));
    for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
      auto const i_end = std::min(bi + TILE, A.extent(0));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
        auto const k_end = std::min(bk + TILE, A.extent(1));

        for (std::size_t j = bj; j < j_end; ++j) {
          for (std::size_t i = bi; i < i_end; ++i) {
            T acc{};
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jki_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
    auto const j_end = std::min(bj + TILE, B.extent(1));
    for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
      auto const k_end = std::min(bk + TILE, A.extent(1));
      for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
        auto const i_end = std::min(bi + TILE, A.extent(0));

        for (std::size_t j = bj; j < j_end; ++j) {
          for (std::size_t k = bk; k < k_end; ++k) {
            T B_kj = B[k, j];
            for (std::size_t i = bi; i < i_end; ++i) {
              C[i, j] += A[i, k] * B_kj;
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kij_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
    auto const k_end = std::min(bk + TILE, A.extent(1));
    for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
      auto const i_end = std::min(bi + TILE, A.extent(0));
      for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
        auto const j_end = std::min(bj + TILE, B.extent(1));

        for (std::size_t k = bk; k < k_end; ++k) {
          for (std::size_t i = bi; i < i_end; ++i) {
            T A_ik = A[i, k];
            for (std::size_t j = bj; j < j_end; ++j) {
              C[i, j] += A_ik * B[k, j];
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kji_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
    auto const k_end = std::min(bk + TILE, A.extent(1));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
      auto const j_end = std::min(bj + TILE, B.extent(1));
      for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
        auto const i_end = std::min(bi + TILE, A.extent(0));

        for (std::size_t k = bk; k < k_end; ++k) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T B_kj = B[k, j];
            for (std::size_t i = bi; i < i_end; ++i) {
              C[i, j] += A[i, k] * B_kj;
            }
          }
        }
      }
    }
  }
}

}

namespace parallel {

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ijk(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
      #pragma omp simd
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ikj(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T A_ik = A[i, k];
      #pragma omp simd
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jik(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T acc{};
      #pragma omp simd
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jki(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t j = 0; j < B.extent(1); ++j) {
    for (std::size_t k = 0; k < A.extent(1); ++k) {
      T B_kj = B[k, j];
      #pragma omp simd
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kij(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t i = 0; i < A.extent(0); ++i) {
      T A_ik = A[i, k];
      #pragma omp simd
      for (std::size_t j = 0; j < B.extent(1); ++j) {
        C[i, j] += A_ik * B[k, j];
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kji(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for collapse(2)
  for (std::size_t k = 0; k < A.extent(1); ++k) {
    for (std::size_t j = 0; j < B.extent(1); ++j) {
      T B_kj = B[k, j];
      #pragma omp simd
      for (std::size_t i = 0; i < A.extent(0); ++i) {
        C[i, j] += A[i, k] * B_kj;
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ijk_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
    auto const i_end = std::min(bi + TILE, A.extent(0));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
      auto const j_end = std::min(bj + TILE, B.extent(1));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
        auto const k_end = std::min(bk + TILE, A.extent(1));

        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T acc{};
            #pragma omp simd
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto ikj_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
    auto const i_end = std::min(bi + TILE, A.extent(0));
    for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
      auto const k_end = std::min(bk + TILE, A.extent(1));
      for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
        auto const j_end = std::min(bj + TILE, B.extent(1));

        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t k = bk; k < k_end; ++k) {
            T A_ik = A[i, k];
            #pragma omp simd
            for (std::size_t j = bj; j < j_end; ++j) {
              C[i, j] += A_ik * B[k, j];
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jik_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
    auto const j_end = std::min(bj + TILE, B.extent(1));
    for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
      auto const i_end = std::min(bi + TILE, A.extent(0));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
        auto const k_end = std::min(bk + TILE, A.extent(1));

        for (std::size_t j = bj; j < j_end; ++j) {
          for (std::size_t i = bi; i < i_end; ++i) {
            T acc{};
            #pragma omp simd
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto jki_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
    auto const j_end = std::min(bj + TILE, B.extent(1));
    for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
      auto const k_end = std::min(bk + TILE, A.extent(1));
      for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
        auto const i_end = std::min(bi + TILE, A.extent(0));

        for (std::size_t j = bj; j < j_end; ++j) {
          for (std::size_t k = bk; k < k_end; ++k) {
            T B_kj = B[k, j];
            #pragma omp simd
            for (std::size_t i = bi; i < i_end; ++i) {
              C[i, j] += A[i, k] * B_kj;
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kij_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
    auto const k_end = std::min(bk + TILE, A.extent(1));
    for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
      auto const i_end = std::min(bi + TILE, A.extent(0));
      for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
        auto const j_end = std::min(bj + TILE, B.extent(1));

        for (std::size_t k = bk; k < k_end; ++k) {
          for (std::size_t i = bi; i < i_end; ++i) {
            T A_ik = A[i, k];
            #pragma omp simd
            for (std::size_t j = bj; j < j_end; ++j) {
              C[i, j] += A_ik * B[k, j];
            }
          }
        }
      }
    }
  }
}

template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto kji_block(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  #pragma omp parallel for
  for (std::size_t bk = 0; bk < A.extent(1); bk += TILE) {
    auto const k_end = std::min(bk + TILE, A.extent(1));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE) {
      auto const j_end = std::min(bj + TILE, B.extent(1));
      for (std::size_t bi = 0; bi < A.extent(0); bi += TILE) {
        auto const i_end = std::min(bi + TILE, A.extent(0));

        for (std::size_t k = bk; k < k_end; ++k) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T B_kj = B[k, j];
            #pragma omp simd
            for (std::size_t i = bi; i < i_end; ++i) {
              C[i, j] += A[i, k] * B_kj;
            }
          }
        }
      }
    }
  }
}
  
}

auto main() -> int {
  auto DIMS = {512zu, 1024zu};

  ankerl::nanobench::Bench bs;
  for (auto const& dims: DIMS) {
    std::vector<double> A(dims * dims, 1.0);
    std::vector<double> B(dims * dims, 1.0);
    std::vector<double> C(dims * dims, 1.0);

    auto mA = std::mdspan(A.data(), dims, dims);
    auto mB = std::mdspan(B.data(), dims, dims);
    auto mC = std::mdspan(C.data(), dims, dims);

    using extents_type = std::dextents<std::size_t, 2>;
    auto mA_tiled = std::mdspan(
      A.data(), layout_right_tiled_right::mapping<extents_type>(extents_type(dims, dims), TILE, TILE)
    );
    auto mB_tiled = std::mdspan(
      B.data(), layout_right_tiled_right::mapping<extents_type>(extents_type(dims, dims), TILE, TILE)
    );
    auto mC_tiled = std::mdspan(
      C.data(), layout_right_tiled_right::mapping<extents_type>(extents_type(dims, dims), TILE, TILE)
    );

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
    bs.title(std::format("parallel {}x{}", dims, dims)).timeUnit(1s, "s").relative(true);
    bs.run("ijk standard", [&]() { parallel::ijk(mA, mB, mC); });
    bs.run("ijk tiled", [&]() { parallel::ijk(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("ijk block standard", [&]() { parallel::ijk_block(mA, mB, mC); });
    bs.run("ijk block tiled", [&]() { parallel::ijk_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("ikj standard", [&]() { parallel::ikj(mA, mB, mC); });
    bs.run("ikj tiled", [&]() { parallel::ikj(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("ikj block standard", [&]() { parallel::ikj_block(mA, mB, mC); });
    bs.run("ikj block tiled", [&]() { parallel::ikj_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("jik standard", [&]() { parallel::jik(mA, mB, mC); });
    bs.run("jik tiled", [&]() { parallel::jik(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("jik block standard", [&]() { parallel::jik_block(mA, mB, mC); });
    bs.run("jik block tiled", [&]() { parallel::jik_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("jki standard", [&]() { parallel::jki(mA, mB, mC); });
    bs.run("jki tiled", [&]() { parallel::jki(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("jki block standard", [&]() { parallel::jki_block(mA, mB, mC); });
    bs.run("jki block tiled", [&]() { parallel::jki_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("kij standard", [&]() { parallel::kij(mA, mB, mC); });
    bs.run("kij tiled", [&]() { parallel::kij(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("kij block standard", [&]() { parallel::kij_block(mA, mB, mC); });
    bs.run("kij block tiled", [&]() { parallel::kij_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("kji standard", [&]() { parallel::kji(mA, mB, mC); });
    bs.run("kji tiled", [&]() { parallel::kji(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("kji block standard", [&]() { parallel::kji_block(mA, mB, mC); });
    bs.run("kji block tiled", [&]() { parallel::kji_block(mA_tiled, mB_tiled, mC_tiled); });
#else
    bs.title(std::format("serial {}x{}", dims, dims)).timeUnit(1s, "s").relative(true);
    bs.run("ijk standard", [&]() { serial::ijk(mA, mB, mC); });
    bs.run("ijk tiled", [&]() { serial::ijk(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("ijk block standard", [&]() { serial::ijk_block(mA, mB, mC); });
    bs.run("ijk block tiled", [&]() { serial::ijk_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("ikj standard", [&]() { serial::ikj(mA, mB, mC); });
    bs.run("ikj tiled", [&]() { serial::ikj(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("ikj block standard", [&]() { serial::ikj_block(mA, mB, mC); });
    bs.run("ikj block tiled", [&]() { serial::ikj_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("jik standard", [&]() { serial::jik(mA, mB, mC); });
    bs.run("jik tiled", [&]() { serial::jik(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("jik block standard", [&]() { serial::jik_block(mA, mB, mC); });
    bs.run("jik block tiled", [&]() { serial::jik_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("jki standard", [&]() { serial::jki(mA, mB, mC); });
    bs.run("jki tiled", [&]() { serial::jki(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("jki block standard", [&]() { serial::jki_block(mA, mB, mC); });
    bs.run("jki block tiled", [&]() { serial::jki_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("kij standard", [&]() { serial::kij(mA, mB, mC); });
    bs.run("kij tiled", [&]() { serial::kij(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("kij block standard", [&]() { serial::kij_block(mA, mB, mC); });
    bs.run("kij block tiled", [&]() { serial::kij_block(mA_tiled, mB_tiled, mC_tiled); });

    bs.run("kji standard", [&]() { serial::kji(mA, mB, mC); });
    bs.run("kji tiled", [&]() { serial::kji(mA_tiled, mB_tiled, mC_tiled); });
    bs.run("kji block standard", [&]() { serial::kji_block(mA, mB, mC); });
    bs.run("kji block tiled", [&]() { serial::kji_block(mA_tiled, mB_tiled, mC_tiled); });
#endif
  }

  return 0;
}
