cmake_minimum_required(VERSION 3.25)

project(layout-tiled LANGUAGES CXX DESCRIPTION "Tiled layouts for std::mdspan" VERSION 0.4.0)

# --- PROJECT OPTIONS --- #
option(
  LayoutTiled_BUILD_TESTS
  "Enable building tests and test infrastructure. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_BUILD_EXAMPLES
  "Enable building examples. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_BUILD_BENCHMARKS
  "Enable building benchmarks. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_INSTALL_PACKAGE
  "Enable installing library as CMake package. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
# --- BENCHMARK OPTIONS --- #
option(
  LayoutTiled_BENCHMARKS_ENABLE_CHECKS
  "Enable checking the results of benchmarks. Default: ON. Values: { ON, OFF }."
  ON
)
option(LayoutTiled_BENCHMARKS_ENABLE_OMP "Enable OpenMP support on benchmarks. Default: ON. Values: { ON, OFF }." ON)
option(
  LayoutTiled_BENCHMARKS_ENABLE_BLAS
  "Enable a BLAS version of the GEMM benchmark. Default: ON. Values: { ON, OFF }."
  ON
)

# --- PROJECT DEPENDENCIES --- #
include(CTest)

# --- PROJECT TARGETS --- #
add_subdirectory(include/layout_tiled)

if(LayoutTiled_BUILD_TESTS)
  add_subdirectory(tests/layout_tiled)
endif()

if(LayoutTiled_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(LayoutTiled_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
