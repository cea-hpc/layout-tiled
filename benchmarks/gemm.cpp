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

#include <mdspan>
#include <print>
#include <vector>

/// Matrix-matrix product: C = A * B
template <
  class T,
  class Extents,
  class Layout,
  class AccessorPolicy
>
auto matrix_product(
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

auto main() -> int {
  constexpr std::size_t DIMS = 512;
  constexpr std::size_t TILE = 64;

  std::vector<double> A(DIMS * DIMS, 1.0);
  std::vector<double> B(DIMS * DIMS, 1.0);
  std::vector<double> C(DIMS * DIMS, 1.0);

  auto mA = std::mdspan(A.data(), DIMS, DIMS);
  auto mB = std::mdspan(B.data(), DIMS, DIMS);
  auto mC = std::mdspan(C.data(), DIMS, DIMS);

  ankerl::nanobench::Bench().run("Standard layout", [&]() { matrix_product(mA, mB, mC); });

  return 0;
}
