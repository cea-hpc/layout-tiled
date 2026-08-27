// include/iai/detail/projected_view.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_PROJECTED_VIEW
#define INCLUDED_IAI_DETAIL_PROJECTED_VIEW

#include <concepts>
#include <cstddef>
#include <experimental/mdspan>
#include <type_traits>
#include <utility>

#include "axis.hpp"

namespace iai::detail {

enum class operand_role {
  input,
  output,
  inout
};

template <class>
inline constexpr bool is_mdspan_v = false;

template <class Element, class Extents, class Layout, class Accessor>
inline constexpr bool is_mdspan_v<std::mdspan<Element, Extents, Layout, Accessor>> = true;

template <class View>
concept mdspan_view = is_mdspan_v<std::remove_cvref_t<View>>;

template <class View, class Axes>
concept valid_binding
  = mdspan_view<View> && (std::remove_cvref_t<View>::rank() == Axes::size) && (Axes::size > 0) && Axes::is_unique();

template <class View, class Identity, class Axes>
concept valid_output_binding
  = valid_binding<View, Axes> && (!std::is_const_v<typename std::remove_cvref_t<View>::element_type>)
    && std::is_constructible_v<typename std::remove_cvref_t<View>::value_type, Identity>
    && std::
      is_assignable_v<typename std::remove_cvref_t<View>::reference, typename std::remove_cvref_t<View>::value_type>;

template <class View, class Axes>
concept valid_inout_binding
  = valid_binding<View, Axes> && (!std::is_const_v<typename std::remove_cvref_t<View>::element_type>)
    && std::is_copy_constructible_v<typename std::remove_cvref_t<View>::value_type>
    && std::
      is_assignable_v<typename std::remove_cvref_t<View>::reference, typename std::remove_cvref_t<View>::value_type>;

template <class View, class Axes>
class projected_input_view {
public:
  using view_type  = View;
  using axes_type  = Axes;
  using value_type = typename view_type::value_type;

  static constexpr bool is_input     = true;
  static constexpr bool is_output    = false;
  static constexpr operand_role role = operand_role::input;

  explicit constexpr projected_input_view(view_type view) : view_(view) {}

  [[nodiscard]] constexpr auto view() const noexcept -> const view_type& {
    return view_;
  }

private:
  view_type view_;
};

template <class View, class Axes, class Identity>
class projected_output_view {
public:
  using view_type     = View;
  using axes_type     = Axes;
  using value_type    = typename view_type::value_type;
  using identity_type = Identity;

  static constexpr bool is_input     = false;
  static constexpr bool is_output    = true;
  static constexpr bool is_inout     = false;
  static constexpr operand_role role = operand_role::output;

  constexpr projected_output_view(view_type view, identity_type identity) :
      view_(view),
      identity_(std::move(identity)) {}

  [[nodiscard]] constexpr auto view() const noexcept -> const view_type& {
    return view_;
  }

  [[nodiscard]] constexpr auto identity() const noexcept -> const identity_type& {
    return identity_;
  }

private:
  view_type view_;
  [[no_unique_address]] identity_type identity_{};
};

template <class View, class Axes>
class projected_inout_view {
public:
  using view_type  = View;
  using axes_type  = Axes;
  using value_type = typename view_type::value_type;

  static constexpr bool is_input     = false;
  static constexpr bool is_output    = true;
  static constexpr bool is_inout     = true;
  static constexpr operand_role role = operand_role::inout;

  explicit constexpr projected_inout_view(view_type view) : view_(view) {}

  [[nodiscard]] constexpr auto view() const noexcept -> const view_type& {
    return view_;
  }

private:
  view_type view_;
};

template <class>
inline constexpr bool is_binding_v = false;

template <class View, class Axes>
inline constexpr bool is_binding_v<projected_input_view<View, Axes>> = true;

template <class View, class Axes, class Identity>
inline constexpr bool is_binding_v<projected_output_view<View, Axes, Identity>> = true;

template <class View, class Axes>
inline constexpr bool is_binding_v<projected_inout_view<View, Axes>> = true;

template <class Type>
inline constexpr bool is_input_binding_v
  = is_binding_v<std::remove_cvref_t<Type>> && std::remove_cvref_t<Type>::is_input;

template <class Type>
inline constexpr bool is_output_binding_v
  = is_binding_v<std::remove_cvref_t<Type>> && std::remove_cvref_t<Type>::is_output;

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_PROJECTED_VIEW
