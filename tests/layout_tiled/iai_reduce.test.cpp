// tests/layout_tiled/iai_reduce.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <experimental/mdspan>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <iai/md_reduce.hpp>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_rtr.hpp>

namespace {

enum : std::size_t {
  I,
  J,
  K,
  L
};

template <class Layout, std::size_t Rank>
using dynamic_mdspan = std::mdspan<double, std::dextents<std::size_t, Rank>, Layout>;

template <class... Arguments>
concept can_reduce_k = requires(Arguments... arguments) { iai::md_reduce<K>(arguments...); };

template <class... Arguments>
concept can_reduce_kk = requires(Arguments... arguments) { iai::md_reduce<K, K>(arguments...); };

template <class View>
concept can_bind_one_axis = requires(View view) { iai::in<I>(view); };

template <class View>
concept can_bind_duplicate_axes = requires(View view) { iai::in<I, I>(view); };

TEST(iai_reduce, gemm_out_and_inout_inner_right) {
  using layout = layout_tiled::layout_rtr<2, 2>;
  std::array<double, 16> a_storage{};
  std::array<double, 16> b_storage{};
  std::array<double, 16> c_storage{};
  auto a = dynamic_mdspan<layout, 2>(a_storage.data(), 4, 4);
  auto b = dynamic_mdspan<layout, 2>(b_storage.data(), 4, 4);
  auto c = dynamic_mdspan<layout, 2>(c_storage.data(), 4, 4);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      a[i, j] = static_cast<double>(2 * i + j + 1);
      b[i, j] = static_cast<double>(i == j ? 2 : 0);
      c[i, j] = -1.0;
    }
  }

  iai::md_reduce<K>(iai::out<I, J>(c, 0.0), iai::in<I, K>(a), iai::in<K, J>(b), [](auto& result, auto lhs, auto rhs) {
    result += lhs * rhs;
  });
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_DOUBLE_EQ((c[i, j]), (2.0 * a[i, j]));
      c[i, j] = 3.0;
    }
  }

  iai::md_reduce<K>(iai::inout<I, J>(c), iai::in<I, K>(a), iai::in<K, J>(b), [](auto& result, auto lhs, auto rhs) {
    result += lhs * rhs;
  });
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_DOUBLE_EQ((c[i, j]), (3.0 + 2.0 * a[i, j]));
    }
  }
}

TEST(iai_reduce, batched_rank_three_contraction) {
  using layout = layout_tiled::layout_rtr<1, 2, 2>;
  std::array<double, 32> a_storage{};
  std::array<double, 32> b_storage{};
  std::array<double, 32> c_storage{};
  auto a = dynamic_mdspan<layout, 3>(a_storage.data(), 2, 4, 4);
  auto b = dynamic_mdspan<layout, 3>(b_storage.data(), 2, 4, 4);
  auto c = dynamic_mdspan<layout, 3>(c_storage.data(), 2, 4, 4);

  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        a[batch, i, j] = static_cast<double>(batch + i + j);
        b[batch, i, j] = static_cast<double>(i == j);
      }
    }
  }

  iai::md_reduce<K>(
    iai::out<I, J, L>(c, 0.0),
    iai::in<I, J, K>(a),
    iai::in<I, K, L>(b),
    [](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
  );
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_DOUBLE_EQ((c[batch, i, j]), (a[batch, i, j]));
      }
    }
  }
}

TEST(iai_reduce, rank_four_with_two_reduction_axes) {
  using output_layout = layout_tiled::layout_rtr<1, 2>;
  using lhs_layout    = layout_tiled::layout_rtr<1, 2, 2>;
  using rhs_layout    = layout_tiled::layout_rtr<2, 2, 2>;
  std::array<double, 8> output_storage{};
  std::array<double, 64> lhs_storage{};
  std::array<double, 64> rhs_storage{};
  auto output = dynamic_mdspan<output_layout, 2>(output_storage.data(), 2, 4);
  auto lhs    = dynamic_mdspan<lhs_layout, 3>(lhs_storage.data(), 2, 4, 4);
  auto rhs    = dynamic_mdspan<rhs_layout, 3>(rhs_storage.data(), 4, 4, 4);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t l = 0; l < 4; ++l) {
        lhs[i, k, l] = static_cast<double>(i + k + l);
      }
    }
  }
  for (std::size_t k = 0; k < 4; ++k) {
    for (std::size_t l = 0; l < 4; ++l) {
      for (std::size_t j = 0; j < 4; ++j) {
        rhs[k, l, j] = static_cast<double>(j + 1);
      }
    }
  }

  iai::md_reduce<K, L>(
    iai::out<I, J>(output, 0.0),
    iai::in<I, K, L>(lhs),
    iai::in<K, L, J>(rhs),
    [](auto& result, auto lhs_value, auto rhs_value) { result += lhs_value * rhs_value; }
  );
  for (std::size_t i = 0; i < 2; ++i) {
    double sum = 0.0;
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t l = 0; l < 4; ++l) {
        sum += lhs[i, k, l];
      }
    }
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_DOUBLE_EQ((output[i, j]), sum * static_cast<double>(j + 1));
    }
  }
}

TEST(iai_reduce, unary_and_heterogeneous_inputs) {
  using output_layout = layout_tiled::layout_rtr<2>;
  using input_layout  = layout_tiled::layout_rtr<2, 2>;
  std::array<double, 4> output_storage{};
  std::array<int, 16> integer_storage{};
  std::array<float, 16> float_storage{};
  auto output   = dynamic_mdspan<output_layout, 1>(output_storage.data(), 4);
  auto integers = std::mdspan<int, std::dextents<std::size_t, 2>, input_layout>(integer_storage.data(), 4, 4);
  auto floats   = std::mdspan<float, std::dextents<std::size_t, 2>, input_layout>(float_storage.data(), 4, 4);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      integers[i, k] = static_cast<int>(i + k);
      floats[i, k]   = 0.5F;
    }
  }

  iai::md_reduce<J>(iai::out<I>(output, 0.0), iai::in<I, J>(integers), [](double& result, int value) {
    result += static_cast<double>(value);
  });
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(output[i], static_cast<double>(4 * i + 6));
  }

  iai::md_reduce<J>(
    iai::out<I>(output, 0.0),
    iai::in<I, J>(integers),
    iai::in<I, J>(floats),
    [](double& result, int integer, float fraction) { result += static_cast<double>(integer) * fraction; }
  );
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(output[i], static_cast<double>(2 * i + 3));
  }
}

TEST(iai_reduce, inner_left_layout_and_non_fragment_aligned_tile_extents) {
  using output_layout = layout_tiled::layout_ltl<3, 2>;
  using lhs_layout    = layout_tiled::layout_ltl<3, 2>;
  using rhs_layout    = layout_tiled::layout_ltl<2, 2>;
  std::array<double, 36> a_storage{};
  std::array<double, 36> b_storage{};
  std::array<double, 36> c_storage{};
  auto a = dynamic_mdspan<lhs_layout, 2>(a_storage.data(), 6, 6);
  auto b = dynamic_mdspan<rhs_layout, 2>(b_storage.data(), 6, 6);
  auto c = dynamic_mdspan<output_layout, 2>(c_storage.data(), 6, 6);
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      a[i, j] = static_cast<double>(i * 6 + j);
      b[i, j] = static_cast<double>(i == j);
    }
  }

  iai::md_reduce<K>(iai::out<I, J>(c, 0.0), iai::in<I, K>(a), iai::in<K, J>(b), [](auto& result, auto lhs, auto rhs) {
    result += lhs * rhs;
  });
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      EXPECT_DOUBLE_EQ((c[i, j]), (a[i, j]));
    }
  }
}

TEST(iai_reduce, broadcast_output_axis_with_strided_input_projection) {
  using output_layout = layout_tiled::layout_rtr<2, 2>;
  using input_layout  = layout_tiled::layout_rtr<2, 2>;
  std::array<double, 16> output_storage{};
  std::array<double, 16> input_storage{};
  auto output = dynamic_mdspan<output_layout, 2>(output_storage.data(), 4, 4);
  auto input  = dynamic_mdspan<input_layout, 2>(input_storage.data(), 4, 4);
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t k = 0; k < 4; ++k) {
      input[j, k] = static_cast<double>(j + k);
    }
  }

  iai::md_reduce<K>(iai::out<I, J>(output, 0.0), iai::in<J, K>(input), [](auto& result, auto value) {
    result += value;
  });
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_DOUBLE_EQ((output[i, j]), static_cast<double>(4 * j + 6));
    }
  }
}

using dynamic_rank_two   = dynamic_mdspan<layout_tiled::layout_rtr<2, 2>, 2>;
using output_ok          = decltype(iai::out<I, J>(std::declval<dynamic_rank_two>(), 0.0));
using lhs_ok             = decltype(iai::in<I, K>(std::declval<dynamic_rank_two>()));
using rhs_ok             = decltype(iai::in<K, J>(std::declval<dynamic_rank_two>()));
using updater_ok         = decltype([](double& result, double lhs, double rhs) { result += lhs * rhs; });
using updater_bad_return = decltype([](double&, double, double) -> int { return 0; });

static_assert(!can_bind_one_axis<dynamic_rank_two>);
static_assert(!can_bind_duplicate_axes<dynamic_rank_two>);
static_assert(can_reduce_k<output_ok, lhs_ok, rhs_ok, updater_ok>);
static_assert(!can_reduce_kk<output_ok, lhs_ok, rhs_ok, updater_ok>);
static_assert(!can_reduce_k<output_ok, lhs_ok, rhs_ok, updater_bad_return>);

using nondense_output = decltype(iai::out<I>(std::declval<dynamic_mdspan<layout_tiled::layout_rtr<2>, 1>>(), 0.0));
using nondense_input  = decltype(iai::in<I, K>(std::declval<dynamic_rank_two>()));
using unary_updater   = decltype([](double& result, double value) { result += value; });
static_assert(!can_reduce_k<nondense_output, nondense_input, unary_updater>);

using reduction_output = decltype(iai::out<I, K>(std::declval<dynamic_rank_two>(), 0.0));
static_assert(!can_reduce_k<reduction_output, lhs_ok, rhs_ok, updater_ok>);

using incompatible_tile_lhs
  = decltype(iai::in<I, K>(std::declval<dynamic_mdspan<layout_tiled::layout_rtr<4, 2>, 2>>()));
static_assert(!can_reduce_k<output_ok, incompatible_tile_lhs, rhs_ok, updater_ok>);

using static_output_view = std::mdspan<double, std::extents<std::size_t, 4, 4>, layout_tiled::layout_rtr<2, 2>>;
using static_lhs_view    = std::mdspan<double, std::extents<std::size_t, 6, 4>, layout_tiled::layout_rtr<2, 2>>;
using static_output      = decltype(iai::out<I, J>(std::declval<static_output_view>(), 0.0));
using static_lhs         = decltype(iai::in<I, K>(std::declval<static_lhs_view>()));
static_assert(!can_reduce_k<static_output, static_lhs, rhs_ok, updater_ok>);

#ifndef NDEBUG
TEST(iai_reduce, rejects_incompatible_dynamic_logical_extents) {
  using layout = layout_tiled::layout_rtr<2, 2>;
  std::array<double, 16> output_storage{};
  std::array<double, 24> lhs_storage{};
  std::array<double, 16> rhs_storage{};
  auto output = dynamic_mdspan<layout, 2>(output_storage.data(), 4, 4);
  auto lhs    = dynamic_mdspan<layout, 2>(lhs_storage.data(), 6, 4);
  auto rhs    = dynamic_mdspan<layout, 2>(rhs_storage.data(), 4, 4);

  EXPECT_DEATH(
    iai::md_reduce<K>(
      iai::out<I, J>(output, 0.0),
      iai::in<I, K>(lhs),
      iai::in<K, J>(rhs),
      [](auto& result, auto lhs_value, auto rhs_value) { result += lhs_value * rhs_value; }
    ),
    "incompatible logical extents"
  );
}
#endif

} // namespace
