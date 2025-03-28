# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

include(GoogleTest)

set(
  ALL_TESTS
  layout_left_tiled_left
  layout_left_tiled_right
  layout_right_tiled_left
  layout_right_tiled_right
)

# We need to force-link GTest targets with libc++ to avoid ABI issues (otherwise defaults to GCC's libstdc++)
target_compile_options(gtest PRIVATE -stdlib=libc++)
target_link_libraries(gtest PRIVATE c++)
target_compile_options(gtest_main PRIVATE -stdlib=libc++)
target_link_libraries(gtest_main PRIVATE c++)

foreach(test ${ALL_TESTS})
  add_executable(layout_tiled.tests.${test})
  target_sources(layout_tiled.tests.${test} PRIVATE ${test}.test.cpp)
  target_compile_options(
    layout_tiled.tests.${test}
    PRIVATE
      -stdlib=libc++
      -Wall
      -Wextra
      -Wshadow
      -Wpedantic
      -Wcast-align
      -Wconversion
      -Wmissing-include-dirs
      -Wformat=2
  )
  target_compile_features(layout_tiled.tests.${test} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.tests.${test}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.tests.${test}
    PRIVATE
      layout_tiled::layout_tiled
      GTest::gtest_main
  )

  gtest_discover_tests(layout_tiled.tests.${test})
endforeach()
