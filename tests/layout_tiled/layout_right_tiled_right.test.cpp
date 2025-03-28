// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#include <layout_tiled/layout_right_tiled_right.hpp>

#include <gtest/gtest.h>

#include <array>
#include <mdspan>

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
