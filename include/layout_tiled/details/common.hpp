/// Copyright (C) 2024, CEA
///
/// Licensed under the MIT License ("the License" hereafter);
/// You may not use this file except in compliance with the License.
/// You should have received a full copy of the License along with this program;
/// if not, you can obtain a copy at:
///
///    https://opensource.org/license/MIT
///
/// The software is provided “as is”, without warranty of any kind, express
/// or implied, including but not limited to the warranties of merchantability,
/// fitness for a particular purpose and noninfringement. In no event shall
/// the authors or copyright holders be liable for any claim, damages or other
/// liability, whether in an action of contract, tort or otherwise, arising from,
/// out of or in connection with the software or the use or other dealings
/// in the software.
///
/// Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

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
constexpr auto is_md_idx_in_span(std::index_sequence<Idxs...>, Extents const& ext, From... values) {
  return (is_idx_in_extent(ext.extent(Idxs), values) && ...);
}

} // namespace details
