// Copyright (C) 2024-2025 CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace details {

template <std::integral IdxType, class From>
  requires (std::integral<From>)
constexpr auto is_idx_in_extent(IdxType extent, From value) -> bool {
  if constexpr (std::is_signed_v<From>) {
    if (value < 0) {
      return false;
    }
  }
  using Tp = std::common_type_t<IdxType, From>;
  return static_cast<Tp>(value) < static_cast<Tp>(extent);
}

template <std::size_t... Idxs, class Extents, class... From>
constexpr auto is_md_idx_in_span(std::index_sequence<Idxs...>, const Extents& ext, From... values) -> bool {
  return (is_idx_in_extent(ext.extent(Idxs), values) && ...);
}

/// If `i > 0`, computes the product of extents up to `i` for all `k` in the range [0, i), otherwise 1.
///
/// Pre-condition: `i < sizeof...(Exts)`
template <size_t... Exts>
constexpr auto fwd_prod_of_extents(size_t i) noexcept -> size_t {
  if (0 == i) {
    return 1;
  }
  constexpr auto extents[] = std::to_array({Exts...});
  std::size_t product      = 1;
  for (size_t k = 0; k < i; ++k) {
    product *= extents[k];
  }
  return product;
}

/// If `i < sizeof...(Exts)`, computes the product of extents up to `i` for all `k` in the range
/// [i+1, sizeof...(Exts)), otherwise 1.
///
/// Pre-condition: `i < sizeof...(Exts)`
template <size_t... Exts>
constexpr auto rev_prod_of_extents(size_t i) noexcept -> size_t {
  if (sizeof...(Exts) < i + 1) {
    return 1;
  }
  constexpr auto extents[] = std::to_array({Exts...});
  std::size_t product      = 1;
  for (size_t k = i + 1; k < sizeof...(Exts); ++k) {
    product *= extents[k];
  }
  return product;
}

} // namespace details
