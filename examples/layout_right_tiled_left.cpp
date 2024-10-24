#include <layout_tiled/layout_right_tiled_left.hpp>

#include <array>
#include <mdspan>
#include <print>

auto main() -> int {
  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using lin_mdspan   = std::mdspan<element_type, extents_type, std::layout_right>;
  using rtl_mdspan   = std::mdspan<element_type, extents_type, layout_right_tiled_left>;
  using rtl_layout   = typename layout_right_tiled_left::template mapping<extents_type>;

  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;
  std::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (row-major over tiles, col-major within tiles)
  std::array<element_type, ROWS * COLS> rtl{
     0,  6,  1,  7,  2,  8,
     3,  9,  4, 10,  5, 11,
    12, 18, 13, 19, 14, 20,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_lin = lin_mdspan(rtl.data(), ROWS, COLS);
  auto m_rtl = rtl_mdspan(rtl.data(), rtl_layout(extents_type(ROWS, COLS), TROWS, TCOLS));

  std::print("\nActual storage (manually tiled):\n");
  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      std::print("{:2d} ", m_lin[i, j]);
    }
    std::print("\n");
  }

  std::print("\nTiled-order iteration (appears linearly stored):\n");
  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      std::print("{:2d} ", m_rtl[i, j]);
    }
    std::print("\n");
  }

  return 0;
}
