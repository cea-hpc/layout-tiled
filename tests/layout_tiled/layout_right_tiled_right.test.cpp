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

#include <layout_tiled/layout_right_tiled_right.hpp>

#include <gtest/gtest.h>

#include <array>
#include <mdspan>

TEST(LayoutRightTiledRight, MapsToLinearStorage) {
  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using lin_mdspan   = std::mdspan<element_type, extents_type, std::layout_right>;
  using rtr_mdspan   = std::mdspan<element_type, extents_type, layout_right_tiled_right<std::size_t, TROWS, TCOLS>>;
  using rtr_layout   = typename layout_right_tiled_right<std::size_t, TROWS, TCOLS>::template mapping<extents_type>;

  // clang-format off
  // Linearly ordered data
  std::array<element_type, ROWS * COLS> lin{
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
  };

  // Manually tiled data (row-major over tiles, row-major within tiles)
  std::array<element_type, ROWS * COLS> rtr{
     0,  1,  2,  6,  7,  8,
     3,  4,  5,  9, 10, 11,
    12, 13, 14, 18, 19, 20,
    15, 16, 17, 21, 22, 23,
  };
  // clang-format on

  auto m_lin = lin_mdspan(lin.data(), ROWS, COLS);
  auto m_rtr = rtr_mdspan(rtr.data(), rtr_layout(extents_type(ROWS, COLS)));

  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      auto lin_val = m_lin[i, j];
      auto rtr_val = m_rtr[i, j];
      EXPECT_EQ(lin_val, rtr_val);
    }
  }
}
