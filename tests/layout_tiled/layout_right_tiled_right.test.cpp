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

// Implement the following tests:
// - ctor.default
// - ctor.extents
// - ctor.mapping
// - required_span_size
// - static_requirements
// - properties
// - index_operator
// - extents
// - tile_extents
// - ntile_in_extents
// - tile_size

// TEST(layout_right_tiled_right, computes_correct_tile_size) {
//   using extents_type = std::dextents<std::size_t, 2>;
//   using layout_rtr   = layout_right_tiled_right<TROWS, TCOLS>::mapping<extents_type>;

//   auto lt = layout_rtr(extents_type{ROWS, COLS});
//   ASSERT_EQ(TROWS * TCOLS, lt.tile_size());
// }

// TEST(layout_right_tiled_right, returns_correct_tile_extent) {
//   using extents_type = std::dextents<std::size_t, 2>;
//   using layout_rtr   = layout_right_tiled_right<TROWS, TCOLS>::mapping<extents_type>;

//   auto lt = layout_rtr(extents_type{ROWS, COLS});
//   EXPECT_EQ(TROWS, lt.textent(0));
//   EXPECT_EQ(TCOLS, lt.textent(1));
// }

// TEST(layout_right_tiled_right, computes_correct_number_of_tiles) {
//   using extents_type = std::dextents<std::size_t, 2>;
//   using layout_rtr   = layout_right_tiled_right<TROWS, TCOLS>::mapping<extents_type>;

//   auto lt = layout_rtr(extents_type{ROWS, COLS});
//   EXPECT_EQ(2, lt.ntiles_in_extent(0));
//   EXPECT_EQ(2, lt.ntiles_in_extent(1));
// }

TEST(layout_right_tiled_right, maps_to_linear_storage) {
  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using layout_rtr   = layout_right_tiled_right<TROWS, TCOLS>::mapping<extents_type>;

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

  auto m_lin = std::mdspan(lin.data(), ROWS, COLS);
  auto m_rtr = std::mdspan(rtr.data(), layout_rtr(extents_type(ROWS, COLS)));

  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      auto lin_val = m_lin[i, j];
      auto rtr_val = m_rtr[i, j];
      EXPECT_EQ(lin_val, rtr_val);
    }
  }
}
