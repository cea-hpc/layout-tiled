// tests/layout_tiled/layout_left_tiled_left.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <experimental/mdspan>

#include <gtest/gtest.h>
#include <layout_tiled/layout_left_tiled_left.hpp>

TEST(layout_left_tiled_left, maps_to_2d_linear_storage) {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<size_t, 2>;
  using layout_ltl   = layout_left_tiled_left<TROWS, TCOLS>;

  // clang-format off
  // Linearly ordered data (row-major)
  std::array<element_type, ROWS * COLS> row{
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
  };

  // Manually tiled data (col-major over tiles, col-major within tiles)
  std::array<element_type, ROWS * COLS> ltl{
     0,  6,  1,  7,  2,  8,
    12, 18, 13, 19, 14, 20,
     3,  9,  4, 10,  5, 11,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_row = std::mdspan<element_type, extents_type>(row.data(), ROWS, COLS);
  auto m_ltl = std::mdspan<element_type, extents_type, layout_ltl>(ltl.data(), ROWS, COLS);

  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      auto row_val = m_row[i, j];
      auto ltl_val = m_ltl[i, j];
      EXPECT_EQ(row_val, ltl_val);
    }
  }
}
