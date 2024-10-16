#include <layout_tiled/layout_right_tiled_right.hpp>

#include <fmt/core.h>

#include <mdspan>

auto main() -> int {
  constexpr std::size_t M  = 8;
  constexpr std::size_t N  = 8;
  constexpr std::size_t TM = 2;
  constexpr std::size_t TN = 2;

  using element_type = int;
  using extents_type = std::dextents<std::size_t, 2>;
  using right_mdspan = std::mdspan<element_type, extents_type, std::layout_right>;
  using tiled_mdspan = std::mdspan<element_type, extents_type, layout_right_tiled_right>;
  using tiled_layout = typename layout_right_tiled_right::template mapping<extents_type>;

  // clang-format off
  element_type tiled[M * N] = {
     0,  1,  4,  5,
     2,  3,  6,  7,
     8,  9, 12, 13,
    10, 11, 14, 15
  };
  element_type right[M * N] = {
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
    12, 13, 14, 15,
  };
  // clang-format on

  auto m_tiled = tiled_mdspan(tiled, tiled_layout(extents_type(M, N), TM, TN));
  auto m_right = right_mdspan(right, M, N);

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      fmt::println("{:02} vs {:02}", m_tiled[i, j], m_right[i, j]);
    }
  }

  return 0;
}
