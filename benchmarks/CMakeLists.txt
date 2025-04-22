# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

set(
  ALL_BENCHMARKS
  gemm
  stencil_2d
)

option(LayoutTiled_ENABLE_OPENMP "Enable building benchmarks with OpenMP support. Values: { ON, OFF }." OFF)
option(LayoutTiled_ENABLE_CHECKS "Enable checking the computed results of benchmarks. Values: { ON, OFF }." OFF)

if(LayoutTiled_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

foreach(bench ${ALL_BENCHMARKS})
  add_executable(layout_tiled.benchmarks.${bench})
  target_sources(layout_tiled.benchmarks.${bench} PRIVATE ${bench}.cpp)
  target_compile_options(
    layout_tiled.benchmarks.${bench}
    PRIVATE
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
      std::mdspan
      fmt::fmt
      nanobench
  )
  if(LayoutTiled_ENABLE_OPENMP)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_ENABLE_OPENMP)
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(LayoutTiled_ENABLE_CHECKS)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_ENABLE_CHECKS)
  endif()
endforeach()
