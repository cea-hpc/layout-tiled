// tests/layout_tiled/iai.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <concepts>
#include <cstddef>
#include <experimental/mdspan>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <iai/detail/access_transform.hpp>
#include <iai/detail/native_tiled_blocked_fragment_reduce.hpp>
#include <iai/detail/native_tiled_cache_block.hpp>
#include <iai/detail/native_tiled_view.hpp>
#include <iai/md_reduce.hpp>
#include <layout_tiled/layout_rtr.hpp>

namespace {

enum : std::size_t {
  I,
  K
};

struct zero_identity {
  constexpr operator double() const noexcept {
    return 0.0;
  }
};

struct weighted_value {
  float value{};
};

struct noncopyable_sum {
  noncopyable_sum()                                          = default;
  noncopyable_sum(const noncopyable_sum&)                    = delete;
  auto operator=(const noncopyable_sum&) -> noncopyable_sum& = delete;

  void operator()(double& result, double value) const {
    result += value;
  }
};

struct tiled_mapping_impostor {
  template <class Extents>
  struct mapping : layout_tiled::layout_rtr<2, 2>::mapping<Extents> {
    using base_type         = typename layout_tiled::layout_rtr<2, 2>::template mapping<Extents>;
    using extents_type      = Extents;
    using layout_type       = layout_tiled::layout_rtr<2, 2>;
    using outer_layout_type = std::layout_right;
    using inner_layout_type = std::layout_right;

    using base_type::base_type;
  };
};

template <class... Arguments>
concept can_md_reduce_k
  = requires(Arguments&&... arguments) { iai::md_reduce<K>(std::forward<Arguments>(arguments)...); };

using recognition_extents = std::dextents<std::size_t, 2>;
using rtr_mapping         = layout_tiled::layout_rtr<2, 4>::mapping<recognition_extents>;
using rtl_mapping         = layout_tiled::layout_rtl<2, 4>::mapping<recognition_extents>;
using ltr_mapping         = layout_tiled::layout_ltr<2, 4>::mapping<recognition_extents>;
using ltl_mapping         = layout_tiled::layout_ltl<2, 4>::mapping<recognition_extents>;
using impostor_mapping    = tiled_mapping_impostor::mapping<recognition_extents>;

static_assert(iai::detail::recognized_layout_tiled_mapping<rtr_mapping>);
static_assert(iai::detail::recognized_layout_tiled_mapping<rtl_mapping>);
static_assert(iai::detail::recognized_layout_tiled_mapping<ltr_mapping>);
static_assert(iai::detail::recognized_layout_tiled_mapping<ltl_mapping>);
static_assert(!iai::detail::recognized_layout_tiled_mapping<impostor_mapping>);
static_assert(iai::detail::layout_tiled_mapping_traits<rtr_mapping>::tile_extents[0] == 2);
static_assert(iai::detail::layout_tiled_mapping_traits<rtr_mapping>::tile_extents[1] == 4);
static_assert(
  std::same_as<typename iai::detail::layout_tiled_mapping_traits<rtl_mapping>::outer_layout_type, std::layout_right>
);
static_assert(
  std::same_as<typename iai::detail::layout_tiled_mapping_traits<rtl_mapping>::inner_layout_type, std::layout_left>
);

using cache_output_view    = std::mdspan<double, std::dextents<std::size_t, 2>, layout_tiled::layout_rtr<2, 4>>;
using cache_lhs_view       = std::mdspan<double, std::dextents<std::size_t, 2>, layout_tiled::layout_rtr<2, 8>>;
using cache_rhs_view       = std::mdspan<double, std::dextents<std::size_t, 2>, layout_tiled::layout_rtr<8, 4>>;
using cache_output_binding = decltype(iai::out<0, 1>(std::declval<cache_output_view>(), 0.0));
using cache_lhs_binding    = decltype(iai::in<0, 2>(std::declval<cache_lhs_view>()));
using cache_rhs_binding    = decltype(iai::in<2, 1>(std::declval<cache_rhs_view>()));
using cache_bindings       = std::tuple<cache_output_binding, cache_lhs_binding, cache_rhs_binding>;

constexpr auto representative_cache_shape = iai::detail::native_tile_block_shape<3>{{2, 3, 4}};
constexpr auto representative_cache_metrics
  = iai::detail::native_tile_block_metrics<3, cache_bindings>(representative_cache_shape);
static_assert(representative_cache_metrics[0] == iai::detail::native_tile_block_operand_metrics{64, 6, 384, 4});
static_assert(representative_cache_metrics[1] == iai::detail::native_tile_block_operand_metrics{128, 8, 1024, 3});
static_assert(representative_cache_metrics[2] == iai::detail::native_tile_block_operand_metrics{256, 12, 3072, 2});
static_assert(iai::detail::native_tile_block_working_set_bytes<3, cache_bindings>(representative_cache_shape) == 4480);
static_assert(iai::detail::native_tile_block_reused_bytes<3, cache_bindings>(representative_cache_shape) == 6272);

constexpr std::array cache_block_candidates{
  iai::detail::native_tile_block_shape<3>{{1, 1, 1}},
  iai::detail::native_tile_block_shape<3>{{2, 2, 1}},
  iai::detail::native_tile_block_shape<3>{{2, 2, 2}},
};
constexpr auto l1_cache_selection
  = iai::detail::select_native_tile_block_shape<3, cache_bindings>(cache_block_candidates, 1200);
constexpr auto larger_cache_selection
  = iai::detail::select_native_tile_block_shape<3, cache_bindings>(cache_block_candidates, 2000);
static_assert(l1_cache_selection.found);
static_assert(l1_cache_selection.shape == cache_block_candidates[1]);
static_assert(l1_cache_selection.working_set_bytes == 1024);
static_assert(larger_cache_selection.found);
static_assert(larger_cache_selection.shape == cache_block_candidates[2]);
static_assert(larger_cache_selection.working_set_bytes == 1792);

using cache_reduction_axes = iai::detail::axis_list<2>;
struct test_native_tiled_schedule_policy {
  static constexpr std::size_t COMPLETE_REDUCTION_BUDGET_BYTES = 800;
  static constexpr std::size_t CACHE_BLOCK_BUDGET_BYTES        = 4096;
};

constexpr auto small_reduction_schedule = iai::detail::select_native_tiled_reduction_schedule_from_tile_counts<
  cache_reduction_axes,
  cache_output_binding,
  std::tuple<cache_lhs_binding, cache_rhs_binding>>(std::array<std::size_t, 3>{2, 3, 2});
constexpr auto large_reduction_schedule = iai::detail::select_native_tiled_reduction_schedule_from_tile_counts<
  cache_reduction_axes,
  cache_output_binding,
  std::tuple<cache_lhs_binding, cache_rhs_binding>>(std::array<std::size_t, 3>{2, 3, 2048});
constexpr auto forced_complete_schedule = iai::detail::select_native_tiled_reduction_schedule_from_tile_counts<
  cache_reduction_axes,
  cache_output_binding,
  std::tuple<cache_lhs_binding, cache_rhs_binding>,
  iai::detail::native_tiled_reduction_override::force_complete_fragment>(std::array<std::size_t, 3>{2, 3, 2048});
constexpr auto forced_blocked_schedule = iai::detail::select_native_tiled_reduction_schedule_from_tile_counts<
  cache_reduction_axes,
  cache_output_binding,
  std::tuple<cache_lhs_binding, cache_rhs_binding>,
  iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(std::array<std::size_t, 3>{2, 3, 2});
constexpr auto policy_overridden_schedule = iai::detail::select_native_tiled_reduction_schedule_from_tile_counts<
  cache_reduction_axes,
  cache_output_binding,
  std::tuple<cache_lhs_binding, cache_rhs_binding>,
  iai::detail::native_tiled_reduction_override::automatic,
  test_native_tiled_schedule_policy>(std::array<std::size_t, 3>{2, 3, 2});
static_assert(small_reduction_schedule.path == iai::detail::native_tiled_reduction_path::complete_fragment);
static_assert(small_reduction_schedule.complete_working_set_bytes == 832);
static_assert(small_reduction_schedule.reduction_tile_count == 2);
static_assert(large_reduction_schedule.path == iai::detail::native_tiled_reduction_path::reduction_blocked_fragment);
static_assert(large_reduction_schedule.complete_working_set_bytes == 786496);
static_assert(large_reduction_schedule.block_shape == iai::detail::native_tile_block_shape<3>{{2, 2, 4}});
static_assert(large_reduction_schedule.block_working_set_bytes == 3328);
static_assert(forced_complete_schedule.path == iai::detail::native_tiled_reduction_path::complete_fragment);
static_assert(forced_blocked_schedule.path == iai::detail::native_tiled_reduction_path::reduction_blocked_fragment);
static_assert(policy_overridden_schedule.path == iai::detail::native_tiled_reduction_path::reduction_blocked_fragment);

using recognized_output_view    = std::mdspan<double, std::dextents<std::size_t, 1>, layout_tiled::layout_rtr<2>>;
using impostor_input_view       = std::mdspan<double, std::dextents<std::size_t, 2>, tiled_mapping_impostor>;
using recognized_output_binding = decltype(iai::out<I>(std::declval<recognized_output_view>(), zero_identity{}));
using impostor_input_binding    = decltype(iai::in<I, K>(std::declval<impostor_input_view>()));

static_assert(
  iai::detail::
    valid_reduce_call<iai::detail::axis_list<K>, recognized_output_binding, impostor_input_binding, noncopyable_sum>()
);
static_assert(!can_md_reduce_k<recognized_output_binding, impostor_input_binding, noncopyable_sum>);

constexpr auto access_transformations_are_composable() -> bool {
  using projection_type = iai::detail::named_axis_projection<3, iai::detail::axis_list<2, 0>>;
  const auto tiler      = iai::detail::rectangular_tiler<3>({8, 16, 24}, {2, 3, 4});
  const auto access     = iai::detail::compose(projection_type{}, tiler);

  return tiler(std::array<std::size_t, 3>{1, 2, 3}) == std::array<std::size_t, 3>{9, 18, 27}
         && access(std::array<std::size_t, 3>{1, 2, 3}) == std::array<std::size_t, 2>{27, 9};
}

constexpr auto affine_modes_are_simplified() -> bool {
  constexpr std::array modes{
    iai::detail::affine_mode{2, 12},
    iai::detail::affine_mode{1, 99},
    iai::detail::affine_mode{5, 0},
    iai::detail::affine_mode{3, 4},
    iai::detail::affine_mode{4, 1},
  };
  constexpr auto simplified = iai::detail::simplify_affine_modes(modes);
  return simplified.size == 1 && simplified[0] == iai::detail::affine_mode{24, 1};
}

static_assert(access_transformations_are_composable());
static_assert(affine_modes_are_simplified());

template <class Layout>
constexpr auto hierarchical_decomposition_matches_mapping() -> bool {
  using extents_type = std::extents<std::size_t, 4, 8>;
  using mapping_type = typename Layout::template mapping<extents_type>;
  const mapping_type mapping{};
  const iai::detail::native_tile_decomposition decomposition(mapping);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      const auto logical = std::array<std::size_t, 2>{i, j};
      if (decomposition(decomposition.split(logical)) != mapping(i, j)) {
        return false;
      }
    }
  }
  return true;
}

static_assert(hierarchical_decomposition_matches_mapping<layout_tiled::layout_rtr<2, 4>>());
static_assert(hierarchical_decomposition_matches_mapping<layout_tiled::layout_rtl<2, 4>>());
static_assert(hierarchical_decomposition_matches_mapping<layout_tiled::layout_ltr<2, 4>>());
static_assert(hierarchical_decomposition_matches_mapping<layout_tiled::layout_ltl<2, 4>>());

template <class Element>
struct offset_counting_accessor {
  using offset_policy    = offset_counting_accessor;
  using element_type     = Element;
  using reference        = Element&;
  using data_handle_type = Element*;

  std::size_t* offset_count{};

  constexpr auto access(data_handle_type handle, std::size_t offset) const noexcept -> reference {
    return handle[offset];
  }

  constexpr auto offset(data_handle_type handle, std::size_t amount) const noexcept -> data_handle_type {
    ++*offset_count;
    return handle + amount;
  }
};

template <class Element>
struct counting_accessor {
  using offset_policy    = counting_accessor;
  using element_type     = Element;
  using reference        = Element&;
  using data_handle_type = Element*;

  std::size_t* access_count{};

  constexpr auto access(data_handle_type handle, std::size_t offset) const noexcept -> reference {
    ++*access_count;
    return handle[offset];
  }

  constexpr auto offset(data_handle_type handle, std::size_t amount) const noexcept -> data_handle_type {
    return handle + amount;
  }
};

template <class Element, class Layout, std::size_t Rank>
using counted_mdspan = std::mdspan<Element, std::dextents<std::size_t, Rank>, Layout, counting_accessor<Element>>;

TEST(iai, bindings_are_lazy_and_md_reduce_is_eager) {
  using output_layout = layout_tiled::layout_rtr<2>;
  using input_layout  = layout_tiled::layout_rtr<2, 2>;

  std::array<double, 4> output_storage{-1.0, -1.0, -1.0, -1.0};
  std::array<double, 16> input_storage{};
  std::size_t access_count = 0;
  using output_view_type   = counted_mdspan<double, output_layout, 1>;
  using input_view_type    = counted_mdspan<double, input_layout, 2>;
  auto output              = output_view_type(
    output_storage.data(),
    typename output_view_type::mapping_type(typename output_view_type::extents_type(4)),
    counting_accessor<double>{&access_count}
  );
  auto input = input_view_type(
    input_storage.data(),
    typename input_view_type::mapping_type(typename input_view_type::extents_type(4, 4)),
    counting_accessor<double>{&access_count}
  );
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      input[i, k] = static_cast<double>(i + k);
    }
  }
  access_count = 0;

  auto output_binding = iai::out<I>(output, zero_identity{});
  auto input_binding  = iai::in<I, K>(input);
  auto inout_binding  = iai::inout<I>(output);

  static_assert(
    std::same_as<decltype(output_binding), iai::projected_output_view<output_view_type, zero_identity, I>>
  );
  static_assert(std::same_as<decltype(input_binding), iai::projected_input_view<input_view_type, I, K>>);
  static_assert(std::same_as<decltype(inout_binding), iai::projected_inout_view<output_view_type, I>>);
  static_assert(std::same_as<typename decltype(input_binding)::view_type, input_view_type>);
  static_assert(std::same_as<typename decltype(output_binding)::identity_type, zero_identity>);
  EXPECT_EQ(access_count, 0U);
  EXPECT_EQ(output_storage, (std::array<double, 4>{-1.0, -1.0, -1.0, -1.0}));
  EXPECT_EQ(output_binding.view().data_handle(), output.data_handle());
  EXPECT_EQ(input_binding.view().data_handle(), input.data_handle());

  noncopyable_sum updater;
  iai::md_reduce<K>(output_binding, input_binding, updater);

  EXPECT_GT(access_count, 0U);
  EXPECT_EQ(output_storage, (std::array<double, 4>{6.0, 10.0, 14.0, 18.0}));
}

TEST(iai, native_tile_slicing_partially_evaluates_the_tile_coordinate_once) {
  using layout_type = layout_tiled::layout_ltl<2, 4>;
  using source_type
    = std::mdspan<double, std::dextents<std::size_t, 2>, layout_type, offset_counting_accessor<double>>;
  using tiled_type = iai::detail::native_tiled_view<source_type>;
  using expected_tile_type
    = std::mdspan<double, std::extents<std::size_t, 2, 4>, std::layout_left, offset_counting_accessor<double>>;
  static_assert(std::same_as<typename tiled_type::tile_type, expected_tile_type>);

  std::array<double, 32> storage{};
  std::size_t offset_count = 0;
  auto source              = source_type(
    storage.data(),
    typename source_type::mapping_type(typename source_type::extents_type(4, 8)),
    offset_counting_accessor<double>{&offset_count}
  );
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      source[i, j] = static_cast<double>(100 * i + j);
    }
  }

  const auto tiled         = tiled_type(source);
  const auto decomposition = tiled.decomposition();
  EXPECT_EQ(decomposition.local_extents(), (std::array<std::size_t, 2>{2, 4}));
  EXPECT_EQ(decomposition.tile_grid_extents(), (std::array<std::size_t, 2>{2, 2}));
  EXPECT_EQ(offset_count, 0);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      const auto logical      = std::array<std::size_t, 2>{i, j};
      const auto hierarchical = decomposition.split(logical);
      EXPECT_EQ(decomposition(hierarchical), source.mapping()(i, j));
    }
  }

  const auto tile = tiled.bind_tile({1, 1});
  EXPECT_EQ(offset_count, 1);
  EXPECT_EQ(tile.data_handle(), storage.data() + source.mapping().tile_offset_outer(2, 4));
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ((tile[i, j]), (source[i + 2, j + 4]));
    }
  }
  EXPECT_EQ(offset_count, 1);
}

TEST(iai, native_tile_ranges_bind_the_origin_once_and_use_affine_outer_strides) {
  using layout_type = layout_tiled::layout_ltl<2, 4>;
  using source_type
    = std::mdspan<double, std::dextents<std::size_t, 2>, layout_type, offset_counting_accessor<double>>;

  std::array<double, 32> storage{};
  std::size_t offset_count = 0;
  auto source              = source_type(
    storage.data(),
    typename source_type::mapping_type(typename source_type::extents_type(4, 8)),
    offset_counting_accessor<double>{&offset_count}
  );
  const auto tiled = iai::detail::native_tiled_view(source);
  const auto range = iai::detail::native_tile_range(tiled, std::array<std::size_t, 2>{0, 0});
  EXPECT_EQ(offset_count, 1);

  const auto origin_tile = range.bind_tile({0, 0});
  EXPECT_EQ(offset_count, 1);
  EXPECT_EQ(origin_tile.data_handle(), storage.data());

  const auto last_tile = range.bind_tile({1, 1});
  EXPECT_EQ(offset_count, 2);
  EXPECT_EQ(last_tile.data_handle(), storage.data() + source.mapping().tile_offset_outer(2, 4));
}

TEST(iai, native_cache_blocks_enumerate_rectangular_tile_groups_and_short_tails) {
  constexpr auto tile_counts  = std::array<std::size_t, 3>{5, 7, 3};
  constexpr auto shape        = iai::detail::native_tile_block_shape<3>{{2, 3, 2}};
  constexpr auto block_counts = iai::detail::native_tile_block_counts(tile_counts, shape);
  static_assert(block_counts == std::array<std::size_t, 3>{3, 3, 2});

  constexpr auto full_block
    = iai::detail::make_native_tile_block(std::array<std::size_t, 3>{1, 1, 0}, tile_counts, shape);
  static_assert(full_block.begin == std::array<std::size_t, 3>{2, 3, 0});
  static_assert(full_block.extents == std::array<std::size_t, 3>{2, 3, 2});
  static_assert(full_block.tile_count() == 12);

  constexpr auto tail_block
    = iai::detail::make_native_tile_block(std::array<std::size_t, 3>{2, 2, 1}, tile_counts, shape);
  static_assert(tail_block.begin == std::array<std::size_t, 3>{4, 6, 2});
  static_assert(tail_block.extents == std::array<std::size_t, 3>{1, 1, 1});

  std::array<std::array<std::size_t, 3>, 4> visited{};
  std::size_t visit_count = 0;
  auto tile_coordinates   = full_block.begin;
  auto record             = [&](const auto& coordinates) {
    visited[visit_count++] = coordinates;
  };
  iai::detail::for_each_native_tile_in_block<0, iai::detail::axis_list<0, 2>>(full_block, tile_coordinates, record);

  EXPECT_EQ(visit_count, 4);
  EXPECT_EQ(
    visited,
    (std::array<std::array<std::size_t, 3>, 4>{
      std::array<std::size_t, 3>{2, 3, 0},
      std::array<std::size_t, 3>{2, 3, 1},
      std::array<std::size_t, 3>{3, 3, 0},
      std::array<std::size_t, 3>{3, 3, 1},
    })
  );
}

TEST(iai, rectangular_native_cache_blocks_own_output_groups_and_reduction_block_tails) {
  enum : std::size_t {
    BlockJ = 1,
    BlockK = 2
  };
  constexpr std::size_t rows       = 6;
  constexpr std::size_t columns    = 12;
  constexpr std::size_t depth      = 24;
  constexpr std::size_t tile_rows  = 2;
  constexpr std::size_t tile_cols  = 4;
  constexpr std::size_t tile_depth = 8;
  using output_type
    = std::mdspan<double, std::dextents<std::size_t, 2>, layout_tiled::layout_rtr<tile_rows, tile_cols>>;
  using lhs_type = std::mdspan<
    double,
    std::dextents<std::size_t, 2>,
    layout_tiled::layout_rtr<tile_rows, tile_depth>,
    offset_counting_accessor<double>>;
  using rhs_type = std::mdspan<
    double,
    std::dextents<std::size_t, 2>,
    layout_tiled::layout_rtr<tile_depth, tile_cols>,
    offset_counting_accessor<double>>;

  std::array<double, rows * columns> output_storage{};
  std::array<double, rows * depth> lhs_storage{};
  std::array<double, depth * columns> rhs_storage{};
  std::size_t input_binding_count = 0;
  auto output                     = output_type(output_storage.data(), rows, columns);
  auto lhs                        = lhs_type(
    lhs_storage.data(),
    typename lhs_type::mapping_type(typename lhs_type::extents_type(rows, depth)),
    offset_counting_accessor<double>{&input_binding_count}
  );
  auto rhs = rhs_type(
    rhs_storage.data(),
    typename rhs_type::mapping_type(typename rhs_type::extents_type(depth, columns)),
    offset_counting_accessor<double>{&input_binding_count}
  );
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < depth; ++k) {
      lhs[i, k] = static_cast<double>(i + k + 1);
    }
  }
  for (std::size_t k = 0; k < depth; ++k) {
    for (std::size_t j = 0; j < columns; ++j) {
      rhs[k, j] = static_cast<double>(k + 2 * j + 1);
    }
  }

  auto output_binding = iai::out<I, BlockJ>(output, 0.0);
  auto inputs         = std::tuple{iai::in<I, BlockK>(lhs), iai::in<BlockK, BlockJ>(rhs)};
  auto updater        = [](double& result, double left, double right) {
    result += left * right;
  };
  constexpr auto block_shape = iai::detail::native_tile_block_shape<3>{{2, 2, 2}};
  iai::detail::host_rectangular_reduction_blocked_native_fragment_reduce<iai::detail::axis_list<BlockK>>(
    output_binding,
    inputs,
    updater,
    block_shape
  );

  // One range-origin binding per operand and current rectangular block, plus
  // residual offsets for non-origin projected tiles. Rebinding every input at
  // every (I,J,K) physical-tile coordinate would require 54 offsets.
  EXPECT_EQ(input_binding_count, 46);

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      double expected = 0.0;
      for (std::size_t k = 0; k < depth; ++k) {
        expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    }
  }
}

TEST(iai, scalar_native_reduce_supports_multiple_axes_broadcasts_mixed_layouts_and_inout) {
  enum : std::size_t {
    ReductionK = 1,
    ReductionL = 2
  };
  using output_type  = std::mdspan<double, std::dextents<std::size_t, 1>, layout_tiled::layout_rtr<2>>;
  using values_type  = std::mdspan<weighted_value, std::dextents<std::size_t, 3>, layout_tiled::layout_ltl<2, 2, 2>>;
  using weights_type = std::mdspan<int, std::dextents<std::size_t, 2>, layout_tiled::layout_rtr<2, 2>>;

  std::array<double, 4> output_storage{};
  std::array<weighted_value, 64> values_storage{};
  std::array<int, 16> weights_storage{};
  auto output  = output_type(output_storage.data(), 4);
  auto values  = values_type(values_storage.data(), 4, 4, 4);
  auto weights = weights_type(weights_storage.data(), 4, 4);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t l = 0; l < 4; ++l) {
        values[i, k, l].value = static_cast<float>(i + k + l);
        weights[k, l]         = static_cast<int>(k + 2 * l + 1);
      }
    }
  }

  const auto updater = [](double& result, weighted_value value, int weight) {
    result += static_cast<double>(value.value) * static_cast<double>(weight);
  };
  auto output_binding  = iai::out<I>(output, 0.0);
  auto values_binding  = iai::in<I, ReductionK, ReductionL>(values);
  auto weights_binding = iai::in<ReductionK, ReductionL>(weights);
  using input_tuple    = std::tuple<decltype(values_binding), decltype(weights_binding)>;
  static_assert(!iai::detail::native_complete_fragment_compatible_v<decltype(output_binding), input_tuple>);
  iai::md_reduce<ReductionK, ReductionL>(output_binding, values_binding, weights_binding, updater);

  std::array<double, 4> expected{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t l = 0; l < 4; ++l) {
        expected[i] += static_cast<double>(values[i, k, l].value) * static_cast<double>(weights[k, l]);
      }
    }
    EXPECT_DOUBLE_EQ(output[i], expected[i]);
    output[i] = 1.5;
  }

  iai::md_reduce<ReductionK, ReductionL>(
    iai::inout<I>(output),
    iai::in<I, ReductionK, ReductionL>(values),
    iai::in<ReductionK, ReductionL>(weights),
    updater
  );
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(output[i], expected[i] + 1.5);
  }
}

TEST(iai, complete_fragment_reduce_stores_each_output_once) {
  enum : std::size_t {
    FragmentJ = 1,
    FragmentK = 2
  };
  constexpr std::size_t rows       = 4;
  constexpr std::size_t nr         = 2 * iai::detail::native_simd_lanes<double>();
  constexpr std::size_t columns    = 2 * nr + 2;
  constexpr std::size_t tile_depth = 4;
  constexpr std::size_t depth      = 2 * tile_depth;
  using output_layout              = layout_tiled::layout_rtr<rows, columns>;
  using lhs_layout                 = layout_tiled::layout_rtr<rows, tile_depth>;
  using rhs_layout                 = layout_tiled::layout_rtr<tile_depth, columns>;
  using output_type = std::mdspan<double, std::dextents<std::size_t, 2>, output_layout, counting_accessor<double>>;
  using lhs_type    = std::mdspan<double, std::dextents<std::size_t, 2>, lhs_layout, offset_counting_accessor<double>>;
  using rhs_type    = std::mdspan<double, std::dextents<std::size_t, 2>, rhs_layout, offset_counting_accessor<double>>;

  std::array<double, rows * columns> output_storage{};
  std::array<double, rows * depth> lhs_storage{};
  std::array<double, depth * columns> rhs_storage{};
  std::size_t output_access_count = 0;
  std::size_t input_binding_count = 0;
  auto output                     = output_type(
    output_storage.data(),
    typename output_type::mapping_type(typename output_type::extents_type(rows, columns)),
    counting_accessor<double>{&output_access_count}
  );
  auto lhs = lhs_type(
    lhs_storage.data(),
    typename lhs_type::mapping_type(typename lhs_type::extents_type(rows, depth)),
    offset_counting_accessor<double>{&input_binding_count}
  );
  auto rhs = rhs_type(
    rhs_storage.data(),
    typename rhs_type::mapping_type(typename rhs_type::extents_type(depth, columns)),
    offset_counting_accessor<double>{&input_binding_count}
  );

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < depth; ++k) {
      lhs[i, k] = static_cast<double>(i + 2 * k + 1);
    }
  }
  for (std::size_t k = 0; k < depth; ++k) {
    for (std::size_t j = 0; j < columns; ++j) {
      rhs[k, j] = static_cast<double>(3 * k + j + 1);
    }
  }

  auto output_binding = iai::out<I, FragmentJ>(output, 0.0);
  auto lhs_binding    = iai::in<I, FragmentK>(lhs);
  auto rhs_binding    = iai::in<FragmentK, FragmentJ>(rhs);
  using input_tuple   = std::tuple<decltype(lhs_binding), decltype(rhs_binding)>;
  static_assert(iai::detail::native_complete_fragment_compatible_v<decltype(output_binding), input_tuple>);
  const auto schedule = iai::detail::select_native_tiled_reduction_schedule<iai::detail::axis_list<FragmentK>>(
    output_binding,
    input_tuple{lhs_binding, rhs_binding}
  );
  EXPECT_EQ(schedule.path, iai::detail::native_tiled_reduction_path::complete_fragment);

  auto updater = [](double& result, double left, double right) {
    result += left * right;
  };
  auto arguments = std::forward_as_tuple(output_binding, lhs_binding, rhs_binding, updater);
  iai::detail::dispatch_native_tiled_reduce<
    iai::detail::axis_list<FragmentK>,
    iai::detail::native_tiled_reduction_override::force_complete_fragment>(arguments, std::index_sequence<0, 1>{});

  EXPECT_EQ(output_access_count, rows * columns);
  constexpr std::size_t input_count          = 2;
  constexpr std::size_t fragment_count       = 3;
  constexpr std::size_t reduction_tile_count = depth / tile_depth;
  EXPECT_EQ(input_binding_count, input_count * (1 + fragment_count * (reduction_tile_count - 1)));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      double expected = 0.0;
      for (std::size_t k = 0; k < depth; ++k) {
        expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    }
  }
}

TEST(iai, reduction_blocked_fragment_reuses_bound_input_tiles_and_carries_output_partials) {
  enum : std::size_t {
    BlockedJ = 1,
    BlockedK = 2
  };
  constexpr std::size_t nr         = 2 * iai::detail::native_simd_lanes<double>();
  constexpr std::size_t rows       = 6;
  constexpr std::size_t columns    = 2 * nr + 2;
  constexpr std::size_t tile_depth = 4;
  constexpr std::size_t depth      = 2 * tile_depth;
  using output_layout              = layout_tiled::layout_rtr<rows, columns>;
  using lhs_layout                 = layout_tiled::layout_rtr<rows, tile_depth>;
  using rhs_layout                 = layout_tiled::layout_rtr<tile_depth, columns>;
  using output_type = std::mdspan<double, std::dextents<std::size_t, 2>, output_layout, counting_accessor<double>>;
  using lhs_type    = std::mdspan<double, std::dextents<std::size_t, 2>, lhs_layout, offset_counting_accessor<double>>;
  using rhs_type    = std::mdspan<double, std::dextents<std::size_t, 2>, rhs_layout, offset_counting_accessor<double>>;

  std::array<double, rows * columns> output_storage{};
  std::array<double, rows * depth> lhs_storage{};
  std::array<double, depth * columns> rhs_storage{};
  std::size_t output_access_count = 0;
  std::size_t input_binding_count = 0;
  auto output                     = output_type(
    output_storage.data(),
    typename output_type::mapping_type(typename output_type::extents_type(rows, columns)),
    counting_accessor<double>{&output_access_count}
  );
  auto lhs = lhs_type(
    lhs_storage.data(),
    typename lhs_type::mapping_type(typename lhs_type::extents_type(rows, depth)),
    offset_counting_accessor<double>{&input_binding_count}
  );
  auto rhs = rhs_type(
    rhs_storage.data(),
    typename rhs_type::mapping_type(typename rhs_type::extents_type(depth, columns)),
    offset_counting_accessor<double>{&input_binding_count}
  );

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < depth; ++k) {
      lhs[i, k] = static_cast<double>(i + k + 1);
    }
  }
  for (std::size_t k = 0; k < depth; ++k) {
    for (std::size_t j = 0; j < columns; ++j) {
      rhs[k, j] = static_cast<double>(2 * k + j + 1);
    }
  }

  auto lhs_binding = iai::in<I, BlockedK>(lhs);
  auto rhs_binding = iai::in<BlockedK, BlockedJ>(rhs);
  auto updater     = [](double& result, double left, double right) {
    result += left * right;
  };
  auto run_blocked = [&](const auto& output_binding) {
    auto arguments = std::forward_as_tuple(output_binding, lhs_binding, rhs_binding, updater);
    iai::detail::dispatch_native_tiled_reduce<
      iai::detail::axis_list<BlockedK>,
      iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
      arguments,
      std::index_sequence<0, 1>{}
    );
  };

  run_blocked(iai::out<I, BlockedJ>(output, 0.0));

  constexpr std::size_t output_elements      = rows * columns;
  constexpr std::size_t reduction_tile_count = depth / tile_depth;
  constexpr std::size_t input_count          = 2;
  EXPECT_EQ(input_binding_count, input_count * reduction_tile_count);
  EXPECT_EQ(output_access_count, output_elements * (1 + 2 * reduction_tile_count));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      double expected = 0.0;
      for (std::size_t k = 0; k < depth; ++k) {
        expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    }
  }

  output_storage.fill(1.5);
  output_access_count = 0;
  input_binding_count = 0;
  run_blocked(iai::inout<I, BlockedJ>(output));

  EXPECT_EQ(input_binding_count, input_count * reduction_tile_count);
  EXPECT_EQ(output_access_count, output_elements * 2 * reduction_tile_count);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      double expected = 1.5;
      for (std::size_t k = 0; k < depth; ++k) {
        expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    }
  }
}

TEST(iai, automatic_schedule_selects_and_executes_reduction_blocked_fragments_for_large_working_sets) {
  enum : std::size_t {
    PolicyJ = 1,
    PolicyK = 2
  };
  constexpr std::size_t rows      = 8;
  constexpr std::size_t columns   = 8;
  constexpr std::size_t depth     = 4096;
  constexpr std::size_t tile_size = 8;
  using matrix_layout             = layout_tiled::layout_rtr<tile_size, tile_size>;
  using matrix_type               = std::mdspan<double, std::dextents<std::size_t, 2>, matrix_layout>;

  std::vector<double> output_storage(rows * columns);
  std::vector<double> lhs_storage(rows * depth);
  std::vector<double> rhs_storage(depth * columns);
  auto output = matrix_type(output_storage.data(), rows, columns);
  auto lhs    = matrix_type(lhs_storage.data(), rows, depth);
  auto rhs    = matrix_type(rhs_storage.data(), depth, columns);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < depth; ++k) {
      lhs[i, k] = static_cast<double>(1 + (i + k) % 7);
    }
  }
  for (std::size_t k = 0; k < depth; ++k) {
    for (std::size_t j = 0; j < columns; ++j) {
      rhs[k, j] = static_cast<double>(1 + (k + 2 * j) % 5);
    }
  }

  auto output_binding = iai::out<I, PolicyJ>(output, 0.0);
  auto lhs_binding    = iai::in<I, PolicyK>(lhs);
  auto rhs_binding    = iai::in<PolicyK, PolicyJ>(rhs);
  const auto inputs   = std::tuple{lhs_binding, rhs_binding};
  const auto schedule
    = iai::detail::select_native_tiled_reduction_schedule<iai::detail::axis_list<PolicyK>>(output_binding, inputs);
  EXPECT_EQ(schedule.path, iai::detail::native_tiled_reduction_path::reduction_blocked_fragment);
  EXPECT_EQ(schedule.complete_working_set_bytes, 524800);
  EXPECT_EQ(schedule.block_shape, (iai::detail::native_tile_block_shape<3>{{1, 1, 4}}));
  EXPECT_EQ(schedule.block_working_set_bytes, 4608);

  iai::md_reduce<PolicyK>(output_binding, lhs_binding, rhs_binding, [](double& result, double left, double right) {
    result += left * right;
  });

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      double expected = 0.0;
      for (std::size_t k = 0; k < depth; ++k) {
        expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    }
  }
}

TEST(iai, semantic_analysis_is_independent_of_storage_mapping) {
  using output_view_type = std::mdspan<double, std::dextents<std::size_t, 1>, std::layout_left>;
  using input_view_type  = std::mdspan<double, std::dextents<std::size_t, 2>, std::layout_right>;

  std::array<double, 2> output_storage{};
  std::array<double, 6> input_storage{};
  auto output         = output_view_type(output_storage.data(), 2);
  auto input          = input_view_type(input_storage.data(), 2, 3);
  auto output_binding = iai::out<I>(output, zero_identity{});
  auto input_binding  = iai::in<I, K>(input);

  using output_binding_type = decltype(output_binding);
  using input_binding_type  = decltype(input_binding);
  using reduction_axes      = iai::detail::axis_list<K>;
  static_assert(
    iai::detail::valid_reduce_call<reduction_axes, output_binding_type, input_binding_type, noncopyable_sum>()
  );
  static_assert(output_binding_type::role == iai::detail::operand_role::output);
  static_assert(input_binding_type::role == iai::detail::operand_role::input);
  static_assert(std::same_as<typename output_binding_type::view_type, output_view_type>);
  static_assert(std::same_as<typename input_binding_type::view_type, input_view_type>);

  const auto bindings = std::tuple{output_binding, input_binding};
  const auto extents  = iai::detail::merge_dynamic_extents<2>(bindings, std::index_sequence<0, 1>{});
  EXPECT_EQ(extents, (std::array<std::size_t, 2>{2, 3}));
  iai::detail::assert_no_aliases(output_binding, std::tuple{input_binding}, std::index_sequence<0>{});

  constexpr auto problem_coordinates = std::array<std::size_t, 2>{1, 2};
  constexpr auto input_coordinates   = iai::detail::project_coordinates<2, input_binding_type>(problem_coordinates);
  static_assert(input_coordinates == std::array<std::size_t, 2>{1, 2});
}

} // namespace
