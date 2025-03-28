// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

namespace details {

template <std::integral IdxType, class From>
  requires(std::integral<From>)
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
constexpr auto is_md_idx_in_span(std::index_sequence<Idxs...>, Extents const& ext, From... values) -> bool {
  return (is_idx_in_extent(ext.extent(Idxs), values) && ...);
}

} // namespace details
