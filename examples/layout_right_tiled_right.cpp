#include <layout_tiled/layout_right_tiled_right.hpp>

#include <array>
#include <mdspan>
#include <print>

auto main() -> int {
  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using lin_mdspan   = std::mdspan<element_type, extents_type, std::layout_right>;
  using rtr_mdspan   = std::mdspan<element_type, extents_type, layout_right_tiled_right>;
  using rtr_layout   = typename layout_right_tiled_right::template mapping<extents_type>;

  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;
  std::print("Array dimensions: {}x{}\nTile dimensions:  {}x{}\n", ROWS, COLS, TROWS, TCOLS);

  // clang-format off
  // Manually tiled data (row-major over tiles, row-major within tiles)
  std::array<element_type, ROWS * COLS> rtr{
     0,  1,  2,  6,  7,  8,
     3,  4,  5,  9, 10, 11,
    12, 13, 14, 18, 19, 20,
    15, 16, 17, 21, 22, 23,
  };
  // clang-format on

  auto m_lin = lin_mdspan(rtr.data(), ROWS, COLS);
  auto m_rtr = rtr_mdspan(rtr.data(), rtr_layout(extents_type(ROWS, COLS), TROWS, TCOLS));

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
      std::print("{:2d} ", m_rtr[i, j]);
    }
    std::print("\n");
  }

  return 0;
}
