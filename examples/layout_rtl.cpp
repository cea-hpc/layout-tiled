// examples/layout_rtl.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef> // size_t
#include <experimental/mdspan>

#include <fmt/core.h>
#include <layout_tiled/layout_rtl.hpp>

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
  auto m_rtl = std::mdspan<element_type, extents_type, layout_tiled::layout_rtl<TROWS, TCOLS>>(rtl.data());

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
