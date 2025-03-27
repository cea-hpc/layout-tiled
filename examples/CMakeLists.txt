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

set(
  ALL_EXAMPLES
  layout_right_tiled_right
  layout_right_tiled_left
  layout_left_tiled_right
  layout_left_tiled_left
)

foreach(example ${ALL_EXAMPLES})
  add_executable(layout_tiled.examples.${example})
  target_sources(layout_tiled.examples.${example} PRIVATE ${example}.cpp)
  target_compile_options(
    layout_tiled.examples.${example}
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
  target_compile_features(layout_tiled.examples.${example} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.examples.${example}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.examples.${example}
    PRIVATE
      layout_tiled::layout_tiled
      fmt::fmt
      c++abi # Need to explicitly link with libc++ for some reason (appears to be related to std::print)
  )
endforeach()
