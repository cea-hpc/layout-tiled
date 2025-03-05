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

set(ALL_BENCHMARKS gemm)

option(LayoutTiled_PARALLEL_BENCH "Run parallel benchmarks" OFF)

# We need to force-link nanobench with libc++ to avoid ABI issues (otherwise defaults to GCC's libstdc++)
target_compile_options(nanobench PRIVATE -stdlib=libc++)
target_link_libraries(nanobench PRIVATE c++)
# Same for fmt
target_compile_options(fmt PRIVATE -stdlib=libc++)
target_link_libraries(fmt PRIVATE c++)

if(LayoutTiled_PARALLEL_BENCH)
  find_package(OpenMP REQUIRED)
endif()

foreach(bench ${ALL_BENCHMARKS})
  add_executable(layout_tiled.benchmarks.${bench})
  target_sources(layout_tiled.benchmarks.${bench} PRIVATE ${bench}.cpp)
  target_compile_options(
    layout_tiled.benchmarks.${bench}
    PRIVATE -stdlib=libc++ -Wall -Wextra -Wpedantic -pedantic -march=native -g3
  )
  target_compile_features(layout_tiled.benchmarks.${bench} PRIVATE cxx_std_23)
  set_target_properties(layout_tiled.benchmarks.${bench} PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      layout_tiled::layout_tiled
      c++ # Need to explicitly link with libc++ for some reason (appears to be related to std::print)
      nanobench
      fmt::fmt
  )
  if(LayoutTiled_PARALLEL_BENCH)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE LAYOUT_TILED_PARALLEL_BENCH)
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()
