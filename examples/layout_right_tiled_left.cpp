// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef> // size_t
#include <layout_tiled/layout_right_tiled_left.hpp>

#include <experimental/mdspan>
#include <fmt/core.h>

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
  auto m_rtl = std::mdspan<element_type, extents_type, layout_right_tiled_left<TROWS, TCOLS>>(rtl.data());

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
