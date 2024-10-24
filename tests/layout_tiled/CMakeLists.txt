# Copyright (C) 2024, CEA
#
# Licensed under the MIT License ("the License" hereafter);
# You may not use this file except in compliance with the License.
# You should have received a full copy of the License along with this program;
# if not, you can obtain a copy at:
#
#    https://opensource.org/license/MIT
#
# The software is provided “as is”, without warranty of any kind, express
# or implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall
# the authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the software or the use or other dealings
# in the software.
#
# Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

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
