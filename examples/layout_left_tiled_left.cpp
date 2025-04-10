// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#include <array>

#include <fmt/core.h>
#include <mdspan/mdspan.hpp>
#include <layout_tiled/layout_left_tiled_left.hpp>

auto main() -> int {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = Kokkos::extents<size_t, ROWS, COLS>;
  fmt::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (col-major over tiles, col-major within tiles)
  std::array<element_type, ROWS * COLS> ltl{
     0,  6,  1,  7,  2,  8,
    12, 18, 13, 19, 14, 20,
     3,  9,  4, 10,  5, 11,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_lin = Kokkos::mdspan<element_type, extents_type>(ltl.data());
  auto m_ltl = Kokkos::mdspan<element_type, extents_type, layout_left_tiled_left<TROWS, TCOLS>>(ltl.data());

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
      fmt::print("{:2d} ", m_ltl[i, j]);
    }
    fmt::print("\n");
  }

  return 0;
}
