// include/iai/detail/access_transform.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_ACCESS_TRANSFORM
#define INCLUDED_IAI_DETAIL_ACCESS_TRANSFORM

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace iai::detail {

/// Projects complete problem coordinates into the dimension order of an
/// operand. Axes omitted by the operand are not represented in the result.
template <std::size_t ProblemRank, class Axes, class ResultCoordinate = std::size_t>
struct named_axis_projection {
  static_assert(Axes::size <= ProblemRank);
  static_assert([] {
    for (const std::size_t axis : Axes::values) {
      if (axis >= ProblemRank) {
        return false;
      }
    }
    return true;
  }());

  template <class Coordinate>
  [[nodiscard]] constexpr auto operator()(const std::array<Coordinate, ProblemRank>& coordinates) const noexcept
    -> std::array<ResultCoordinate, Axes::size> {
    return project(coordinates, std::make_index_sequence<Axes::size>());
  }

private:
  template <class Coordinate, std::size_t... Dimensions>
  [[nodiscard]] static constexpr auto
  project(const std::array<Coordinate, ProblemRank>& coordinates, std::index_sequence<Dimensions...>) noexcept
    -> std::array<ResultCoordinate, Axes::size> {
    return {static_cast<ResultCoordinate>(coordinates[Axes::values[Dimensions]])...};
  }
};

/// Maps local coordinates of a rectangular execution partition back into the
/// complete problem index space. This describes iteration only; it owns no
/// element storage.
template <std::size_t Rank, class Index = std::size_t>
class rectangular_tiler {
public:
  using index_type      = Index;
  using coordinate_type = std::array<index_type, Rank>;

  constexpr rectangular_tiler(coordinate_type origin, coordinate_type extents) noexcept :
      origin_(origin),
      extents_(extents) {
    for (const index_type extent : extents_) {
      assert(extent > index_type{0});
    }
  }

  [[nodiscard]] constexpr auto origin() const noexcept -> const coordinate_type& {
    return origin_;
  }

  [[nodiscard]] constexpr auto extents() const noexcept -> const coordinate_type& {
    return extents_;
  }

  template <class Coordinate>
  [[nodiscard]] constexpr auto operator()(const std::array<Coordinate, Rank>& local) const noexcept
    -> coordinate_type {
    coordinate_type result{};
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      const index_type coordinate = static_cast<index_type>(local[axis]);
      assert(coordinate < extents_[axis]);
      result[axis] = origin_[axis] + coordinate;
    }
    return result;
  }

private:
  coordinate_type origin_{};
  coordinate_type extents_{};
};

/// Restricted function composition used for coordinate transformations.
/// `compose(outer, inner)(x)` evaluates `outer(inner(x))`.
template <class Outer, class Inner>
class coordinate_transform_composition {
public:
  constexpr coordinate_transform_composition(Outer outer, Inner inner) noexcept(
    std::is_nothrow_move_constructible_v<Outer> && std::is_nothrow_move_constructible_v<Inner>
  ) :
      outer_(std::move(outer)),
      inner_(std::move(inner)) {}

  template <class Coordinate>
  [[nodiscard]] constexpr decltype(auto)
  operator()(const Coordinate& coordinate) const noexcept(noexcept(outer_(inner_(coordinate)))) {
    return outer_(inner_(coordinate));
  }

private:
  [[no_unique_address]] Outer outer_;
  [[no_unique_address]] Inner inner_;
};

template <class Outer, class Inner>
[[nodiscard]] constexpr auto compose(Outer outer, Inner inner) {
  return coordinate_transform_composition<Outer, Inner>(std::move(outer), std::move(inner));
}

/// One affine access mode. Mode sequences passed to `simplify_affine_modes`
/// are ordered from the slowest-varying to the fastest-varying mode.
struct affine_mode {
  std::size_t extent{};
  std::size_t stride{};

  constexpr auto operator==(const affine_mode&) const -> bool = default;
};

template <std::size_t Capacity>
struct affine_mode_sequence {
  std::array<affine_mode, Capacity> modes{};
  std::size_t size{};

  [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept -> const affine_mode& {
    assert(index < size);
    return modes[index];
  }

  constexpr auto operator==(const affine_mode_sequence&) const -> bool = default;
};

/// Removes extent-one and broadcast modes, then coalesces adjacent modes whose
/// strides prove that they form one contiguous affine mode. Iteration extents
/// remain the responsibility of the execution partition.
template <std::size_t Rank>
[[nodiscard]] constexpr auto simplify_affine_modes(const std::array<affine_mode, Rank>& modes) noexcept
  -> affine_mode_sequence<Rank> {
  affine_mode_sequence<Rank> result{};
  for (const affine_mode mode : modes) {
    if (mode.extent == 1 || mode.stride == 0) {
      continue;
    }

    if (result.size > 0) {
      affine_mode& outer = result.modes[result.size - 1];
      const bool stride_product_is_representable
        = mode.extent > 0 && mode.extent <= std::numeric_limits<std::size_t>::max() / mode.stride;
      if (stride_product_is_representable && outer.stride == mode.extent * mode.stride
          && outer.extent <= std::numeric_limits<std::size_t>::max() / mode.extent) {
        outer.extent *= mode.extent;
        outer.stride = mode.stride;
        continue;
      }
    }

    result.modes[result.size] = mode;
    ++result.size;
  }
  return result;
}

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_ACCESS_TRANSFORM
