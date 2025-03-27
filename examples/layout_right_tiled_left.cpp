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

#include <array>
#include <mdspan>

#include <fmt/core.h>
#include <layout_tiled/layout_right_tiled_left.hpp>

auto main() -> int {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::extents<size_t, ROWS, COLS>;
  fmt::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (row-major over tiles, col-major within tiles)
  std::array<element_type, ROWS * COLS> rtl{
     0,  6,  1,  7,  2,  8,
     3,  9,  4, 10,  5, 11,
    12, 18, 13, 19, 14, 20,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_lin = std::mdspan<element_type, extents_type>(rtl.data());
  auto m_rtl = std::mdspan<element_type, extents_type, layout_right_tiled_left<TROWS, TCOLS>>(rtl.data());

  fmt::print("\nActual storage (manually tiled):\n");
  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      fmt::print("{:2d} ", m_lin[i, j]);
    }
    fmt::print("\n");
  }

  fmt::print("\nTiled-order iteration (appears linearly stored):\n");
  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      fmt::print("{:2d} ", m_rtl[i, j]);
    }
    fmt::print("\n");
  }

  return 0;
}
