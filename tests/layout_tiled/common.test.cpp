// tests/layout_tiled/common.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <experimental/mdspan>

#include <gtest/gtest.h>
#include <layout_tiled/detail/common.hpp>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_ltr.hpp>
#include <layout_tiled/layout_rtl.hpp>
#include <layout_tiled/layout_rtr.hpp>

namespace {

template <template <size_t...> class Layout>
auto required_span_size_of_large_empty_mapping() -> int {
  using extents_type = std::dextents<int, 3>;
  using mapping_type = typename Layout<1, 1, 1>::template mapping<extents_type>;
  return mapping_type(extents_type(50'000, 50'000, 0)).required_span_size();
}

template <class Layout, int Rows, int Cols>
constexpr auto mapping_is_a_2d_permutation() -> bool {
  using extents_type = std::dextents<int, 2>;
  using mapping_type = typename Layout::template mapping<extents_type>;

  const mapping_type mapping(extents_type(Rows, Cols));
  std::array<bool, static_cast<size_t>(Rows * Cols)> seen{};
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      const int offset = mapping(i, j);
      if (offset < 0 || offset >= Rows * Cols || seen[static_cast<size_t>(offset)]) {
        return false;
      }
      seen[static_cast<size_t>(offset)] = true;
    }
  }
  return mapping.required_span_size() == Rows * Cols;
}

template <template <size_t...> class Layout>
constexpr auto rank_four_mapping_is_a_permutation() -> bool {
  using extents_type = std::dextents<int, 4>;
  using mapping_type = typename Layout<1, 2, 3, 4>::template mapping<extents_type>;

  const mapping_type mapping(extents_type(2, 4, 6, 8));
  std::array<bool, 384> seen{};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 6; ++k) {
        for (int l = 0; l < 8; ++l) {
          const int offset = mapping(i, j, k, l);
          if (offset < 0 || offset >= 384 || seen[static_cast<size_t>(offset)]) {
            return false;
          }
          seen[static_cast<size_t>(offset)] = true;
        }
      }
    }
  }
  return mapping.required_span_size() == 384;
}

template <template <size_t...> class Layout>
constexpr auto broadened_indexing_cases_are_permutations() -> bool {
  return mapping_is_a_2d_permutation<Layout<1, 1>, 3, 5>() && mapping_is_a_2d_permutation<Layout<3, 5>, 3, 5>()
         && rank_four_mapping_is_a_permutation<Layout>();
}

static_assert(broadened_indexing_cases_are_permutations<layout_tiled::layout_ltl>());
static_assert(broadened_indexing_cases_are_permutations<layout_tiled::layout_ltr>());
static_assert(broadened_indexing_cases_are_permutations<layout_tiled::layout_rtl>());
static_assert(broadened_indexing_cases_are_permutations<layout_tiled::layout_rtr>());

#if !defined(NDEBUG)
template <template <size_t...> class Layout>
auto convert_to_unrepresentable_span_size() -> void {
  using source_extents_type = std::dextents<long long, 2>;
  using source_mapping_type = typename Layout<1, 1>::template mapping<source_extents_type>;
  using dest_extents_type   = std::dextents<int, 2>;
  using dest_mapping_type   = typename Layout<1, 1>::template mapping<dest_extents_type>;

  source_mapping_type source(source_extents_type(50'000, 50'000));
  static_cast<void>(dest_mapping_type(source));
}

template <template <size_t...> class Layout>
auto construct_unrepresentable_span_size() -> void {
  using extents_type = std::dextents<int, 2>;
  using mapping_type = typename Layout<1, 1>::template mapping<extents_type>;
  static_cast<void>(mapping_type(extents_type(50'000, 50'000)));
}

template <template <size_t...> class Layout>
auto construct_nondivisible_extents() -> void {
  using extents_type = std::dextents<int, 2>;
  using mapping_type = typename Layout<2, 3>::template mapping<extents_type>;
  static_cast<void>(mapping_type(extents_type(5, 6)));
}
#endif

} // namespace

TEST(common, zero_extent_prevents_span_size_overflow) {
  EXPECT_EQ(required_span_size_of_large_empty_mapping<layout_tiled::layout_ltl>(), 0);
  EXPECT_EQ(required_span_size_of_large_empty_mapping<layout_tiled::layout_ltr>(), 0);
  EXPECT_EQ(required_span_size_of_large_empty_mapping<layout_tiled::layout_rtl>(), 0);
  EXPECT_EQ(required_span_size_of_large_empty_mapping<layout_tiled::layout_rtr>(), 0);
}

#if !defined(NDEBUG)
TEST(common, converting_constructor_rejects_unrepresentable_span_size) {
  EXPECT_DEATH(
    convert_to_unrepresentable_span_size<layout_tiled::layout_ltl>(),
    "required_span_size_is_representable"
  );
  EXPECT_DEATH(
    convert_to_unrepresentable_span_size<layout_tiled::layout_ltr>(),
    "required_span_size_is_representable"
  );
  EXPECT_DEATH(
    convert_to_unrepresentable_span_size<layout_tiled::layout_rtl>(),
    "required_span_size_is_representable"
  );
  EXPECT_DEATH(
    convert_to_unrepresentable_span_size<layout_tiled::layout_rtr>(),
    "required_span_size_is_representable"
  );
}

TEST(common, direct_constructor_rejects_unrepresentable_span_size) {
  EXPECT_DEATH(construct_unrepresentable_span_size<layout_tiled::layout_ltl>(), "required_span_size_is_representable");
  EXPECT_DEATH(construct_unrepresentable_span_size<layout_tiled::layout_ltr>(), "required_span_size_is_representable");
  EXPECT_DEATH(construct_unrepresentable_span_size<layout_tiled::layout_rtl>(), "required_span_size_is_representable");
  EXPECT_DEATH(construct_unrepresentable_span_size<layout_tiled::layout_rtr>(), "required_span_size_is_representable");
}

TEST(common, direct_constructor_rejects_nondivisible_extents) {
  EXPECT_DEATH(construct_nondivisible_extents<layout_tiled::layout_ltl>(), "tile_extents_are_multiples_of_extents");
  EXPECT_DEATH(construct_nondivisible_extents<layout_tiled::layout_ltr>(), "tile_extents_are_multiples_of_extents");
  EXPECT_DEATH(construct_nondivisible_extents<layout_tiled::layout_rtl>(), "tile_extents_are_multiples_of_extents");
  EXPECT_DEATH(construct_nondivisible_extents<layout_tiled::layout_rtr>(), "tile_extents_are_multiples_of_extents");
}
#endif
