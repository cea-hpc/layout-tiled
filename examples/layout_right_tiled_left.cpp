#include <layout_tiled/layout_right_tiled_left.hpp>

#include <cstdio>
#include <mdspan>

auto main() -> int {
  constexpr std::size_t ROWS  = 4;
  constexpr std::size_t COLS  = 6;

  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using lin_mdspan   = std::mdspan<element_type, extents_type, std::layout_right>;
  using rtl_mdspan   = std::mdspan<element_type, extents_type, layout_right_tiled_left>;
  using rtl_layout   = typename layout_right_tiled_left::template mapping<extents_type>;

  // clang-format off
  // Linearly ordered data
  element_type lin[ROWS * COLS] = {
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
  };

  // Manually tiled data (row-major over tiles, row-major within tiles)
  constexpr std::size_t TROWS = 2;
  constexpr std::size_t TCOLS = 3;
  element_type rtl[ROWS * COLS] = {
     0,  6,  1,  7,  2,  8,
     3,  9,  4, 10,  5, 11,
    12, 18, 13, 19, 14, 20,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_lin = lin_mdspan(lin, ROWS, COLS);
  auto m_rtl = rtl_mdspan(rtl, rtl_layout(extents_type(ROWS, COLS), TROWS, TCOLS));

  int errs = 0;
  for (std::size_t i = 0; i < ROWS; ++i) {
    for (std::size_t j = 0; j < COLS; ++j) {
      if (m_lin[i, j] != m_rtl[i, j]) {
        ++errs;
        printf(
          "Mismatch at (%zu, %zu): lin[%zu, %zu] = %d but rtl[%zu, %zu] = %d\n",
          i, j,
          i, j, m_lin[i, j],
          i, j, m_rtl[i, j]
        );
      }
    }
  }

  if (errs != 0) {
    fprintf(stderr, "\nErrors: %d\n", errs);
  } else {
    printf("Success!\n");
  }

  return errs;
}
