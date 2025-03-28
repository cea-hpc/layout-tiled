# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

set(ALL_BENCHMARKS gemm)

option(LayoutTiled_BENCH_OPENMP "Run benchmarks in parallel using OpenMP" OFF)

# We need to force-link nanobench with libc++ to avoid ABI issues (otherwise defaults to GCC's libstdc++)
target_compile_options(nanobench PRIVATE -stdlib=libc++)
target_link_libraries(nanobench PRIVATE c++abi)
# Same for {fmt}
target_compile_options(fmt PRIVATE -stdlib=libc++)
target_link_libraries(fmt PRIVATE c++abi)

if(LayoutTiled_BENCH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

foreach(bench ${ALL_BENCHMARKS})
  add_executable(layout_tiled.benchmarks.${bench})
  target_sources(layout_tiled.benchmarks.${bench} PRIVATE ${bench}.cpp)
  target_compile_options(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      -stdlib=libc++
      -Wall
      -Wextra
      -Wpedantic
      -pedantic
      -march=native
      -g3
  )
  target_compile_features(layout_tiled.benchmarks.${bench} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.benchmarks.${bench}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      layout_tiled::layout_tiled
      c++ # Need to explicitly link with libc++ for some reason (appears to be related to std::print)
      nanobench
      fmt::fmt
  )
  if(LayoutTiled_BENCH_OPENMP)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE LAYOUT_TILED_BENCH_OPENMP)
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()
