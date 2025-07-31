// tests/layout_tiled/layout_rtr.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <experimental/mdspan>

#include <gtest/gtest.h>
#include <layout_tiled/layout_rtr.hpp>

TEST(layout_rtr, maps_to_2d_linear_storage) {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<size_t, 2>;
  using layout_rtr   = layout_tiled::layout_rtr<TROWS, TCOLS>;

  // clang-format off
  // Linearly ordered data (row-major)
  std::array<element_type, ROWS * COLS> row{
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

  auto m_row = std::mdspan<element_type, extents_type>(row.data(), ROWS, COLS);
  auto m_rtr = std::mdspan<element_type, extents_type, layout_rtr>(rtr.data(), ROWS, COLS);

  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      auto row_val = m_row[i, j];
      auto rtr_val = m_rtr[i, j];
      EXPECT_EQ(row_val, rtr_val);
    }
  }
}
