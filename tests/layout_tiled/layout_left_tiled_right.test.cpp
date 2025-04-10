// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#include <array>

#include <gtest/gtest.h>
#include <mdspan/mdspan.hpp>

#include <layout_tiled/layout_left_tiled_right.hpp>

TEST(layout_left_tiled_right, maps_to_linear_storage) {
  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;

  using element_type = int;
  using extents_type = Kokkos::dextents<std::size_t, 2>;
  using layout_ltr   = layout_left_tiled_right<TROWS, TCOLS>;

  // clang-format off
  // Linearly ordered data
  std::array<element_type, ROWS * COLS> lin{
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
  };

  // Manually tiled data (row-major over tiles, row-major within tiles)
  std::array<element_type, ROWS * COLS> ltr{
     0,  1,  2,  6,  7,  8,
    12, 13, 14, 18, 19, 20,
     3,  4,  5,  9, 10, 11,
    15, 16, 17, 21, 22, 23,
  };
  // clang-format on

  auto m_lin = Kokkos::mdspan<element_type, extents_type>(lin.data(), ROWS, COLS);
  auto m_ltr = Kokkos::mdspan<element_type, extents_type, layout_ltr>(ltr.data(), ROWS, COLS);

  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      auto lin_val = m_lin[i, j];
      auto ltr_val = m_ltr[i, j];
      EXPECT_EQ(lin_val, ltr_val);
    }
  }
}
