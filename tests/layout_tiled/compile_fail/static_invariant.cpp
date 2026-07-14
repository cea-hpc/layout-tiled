// tests/layout_tiled/compile_fail/static_invariant.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <experimental/mdspan>

#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_ltr.hpp>
#include <layout_tiled/layout_rtl.hpp>
#include <layout_tiled/layout_rtr.hpp>

#if LAYOUT_TILED_TEST_LAYOUT == 0
#define LAYOUT_TILED_TEST_POLICY layout_tiled::layout_ltl
#elif LAYOUT_TILED_TEST_LAYOUT == 1
#define LAYOUT_TILED_TEST_POLICY layout_tiled::layout_ltr
#elif LAYOUT_TILED_TEST_LAYOUT == 2
#define LAYOUT_TILED_TEST_POLICY layout_tiled::layout_rtl
#elif LAYOUT_TILED_TEST_LAYOUT == 3
#define LAYOUT_TILED_TEST_POLICY layout_tiled::layout_rtr
#else
#error "Unknown layout selected for static invariant test"
#endif

#if LAYOUT_TILED_TEST_INVARIANT == 0
using mapping_type = LAYOUT_TILED_TEST_POLICY<2, 3>::mapping<std::extents<int, 5, 6>>;
#elif LAYOUT_TILED_TEST_INVARIANT == 1
using mapping_type = LAYOUT_TILED_TEST_POLICY<1, 1>::mapping<std::extents<int, 50'000, 50'000>>;
#elif LAYOUT_TILED_TEST_INVARIANT == 2
using mapping_type = LAYOUT_TILED_TEST_POLICY<50'000, 50'000>::mapping<std::extents<int, 0, 50'000>>;
#else
#error "Unknown invariant selected for static invariant test"
#endif

auto main() -> int {
  static_cast<void>(sizeof(mapping_type));
  return 0;
}
