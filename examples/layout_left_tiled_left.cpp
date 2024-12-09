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

#include <array>
#include <mdspan>
#include <print>

auto main() -> int {
  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using ltl_layout   = layout_left_tiled_left<TROWS, TCOLS>::mapping<extents_type>;
  std::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (col-major over tiles, col-major within tiles)
  std::array<element_type, ROWS * COLS> ltl{
     0,  6,  1,  7,  2,  8,
    12, 18, 13, 19, 14, 20,
     3,  9,  4, 10,  5, 11,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_lin = std::mdspan(ltl.data(), ROWS, COLS);
  auto m_ltl = std::mdspan(ltl.data(), ltl_layout(extents_type(ROWS, COLS)));

  std::print("\nActual storage (manually tiled):\n");
  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      std::print("{:2d} ", m_lin[i, j]);
    }
    std::print("\n");
  }

  std::print("\nTiled-order iteration (appears linearly stored):\n");
  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      std::print("{:2d} ", m_ltl[i, j]);
    }
    std::print("\n");
  }

  return 0;
}
