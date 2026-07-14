// tests/layout_tiled/common.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <experimental/mdspan>
#include <type_traits>

#include <gtest/gtest.h>
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

template <template <size_t...> class Layout>
auto check_mapping_conversions() -> void {
  using dynamic_extents_type      = std::dextents<int, 2>;
  using long_extents_type         = std::dextents<long long, 2>;
  using static_extents_type       = std::extents<int, 4, 6>;
  using mixed_extents_type        = std::extents<int, 4, std::dynamic_extent>;
  using incompatible_extents_type = std::extents<int, 8, 6>;

  using dynamic_mapping_type      = typename Layout<2, 3>::template mapping<dynamic_extents_type>;
  using long_mapping_type         = typename Layout<2, 3>::template mapping<long_extents_type>;
  using static_mapping_type       = typename Layout<2, 3>::template mapping<static_extents_type>;
  using mixed_mapping_type        = typename Layout<2, 3>::template mapping<mixed_extents_type>;
  using incompatible_mapping_type = typename Layout<2, 3>::template mapping<incompatible_extents_type>;

  static_assert(std::is_constructible_v<dynamic_mapping_type, static_mapping_type>);
  static_assert(std::is_convertible_v<static_mapping_type, dynamic_mapping_type>);
  static_assert(std::is_constructible_v<static_mapping_type, dynamic_mapping_type>);
  static_assert(!std::is_convertible_v<dynamic_mapping_type, static_mapping_type>);
  static_assert(std::is_constructible_v<mixed_mapping_type, static_mapping_type>);
  static_assert(!std::is_constructible_v<static_mapping_type, incompatible_mapping_type>);
  static_assert(std::is_constructible_v<dynamic_mapping_type, long_mapping_type>);

  const static_mapping_type static_mapping;
  const dynamic_mapping_type dynamic_mapping(static_mapping);
  const mixed_mapping_type mixed_mapping(static_mapping);
  const static_mapping_type static_from_dynamic(dynamic_mapping);
  const long_mapping_type long_mapping(long_extents_type(4, 6));
  const dynamic_mapping_type narrowed_mapping(long_mapping);

  EXPECT_EQ(dynamic_mapping.extents(), dynamic_extents_type(4, 6));
  EXPECT_EQ(mixed_mapping.extents(), mixed_extents_type(6));
  EXPECT_EQ(static_from_dynamic.extents(), static_extents_type());
  EXPECT_EQ(narrowed_mapping.extents(), dynamic_extents_type(4, 6));
}

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

TEST(common, mapping_converting_constructors) {
  check_mapping_conversions<layout_tiled::layout_ltl>();
  check_mapping_conversions<layout_tiled::layout_ltr>();
  check_mapping_conversions<layout_tiled::layout_rtl>();
  check_mapping_conversions<layout_tiled::layout_rtr>();
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
