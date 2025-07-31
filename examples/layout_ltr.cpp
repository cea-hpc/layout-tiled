// examples/layout_ltr.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef> // size_t
#include <experimental/mdspan>

#include <fmt/core.h>
#include <layout_tiled/layout_ltr.hpp>

auto main() -> int {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::extents<size_t, ROWS, COLS>;
  fmt::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (col-major over tiles, row-major within tiles)
  std::array<element_type, ROWS * COLS> ltr{
     0,  1,  2,  6,  7,  8,
    12, 13, 14, 18, 19, 20,
     3,  4,  5,  9, 10, 11,
    15, 16, 17, 21, 22, 23,
  };
  // clang-format on

  auto m_lin = std::mdspan<element_type, extents_type>(ltr.data());
  auto m_ltr = std::mdspan<element_type, extents_type, layout_tiled::layout_ltr<TROWS, TCOLS>>(ltr.data());

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
      fmt::print("{:2d} ", m_ltr[i, j]);
    }
    fmt::print("\n");
  }

  return 0;
}
