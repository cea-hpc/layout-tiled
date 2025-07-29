cmake_minimum_required(VERSION 3.25) # support for C++26

project(layout-tiled VERSION 0.3.0 LANGUAGES CXX DESCRIPTION "Tiled layouts for C++23 std::mdspan")

# --- PROJECT OPTIONS --- #
option(LayoutTiled_BUILD_TESTS "Enable building tests. Values: { ON, OFF}." OFF)
option(LayoutTiled_BUILD_EXAMPLES "Enable building examples. Values: { ON, OFF }." OFF)
option(LayoutTiled_BUILD_BENCHMARKS "Enable building benchmarks. Values: { ON, OFF }." OFF)

option(LayoutTiled_ENABLE_DEVICE_OFFLOAD "Enable support for offloading to devices (GPUs). Values: { ON, OFF }." OFF)

# --- PROJECT DEPENDENCIES --- #
include(FetchContent)
FetchContent_Declare(mdspan GIT_REPOSITORY https://github.com/kokkos/mdspan.git GIT_TAG origin/stable)
FetchContent_MakeAvailable(mdspan)
target_compile_definitions(mdspan INTERFACE -DMDSPAN_USE_BRACKET_OPERATOR)

if(LayoutTiled_ENABLE_DEVICE_OFFLOAD)
  message(STATUS "Enabling support for device offload:")
  include(CheckLanguage)
  check_language(CUDA)
  check_language(HIP)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "Using ${CMAKE_CXX_STANDARD} as CMAKE_CUDA_STANDARD")
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
  elseif(CMAKE_HIP_COMPILER)
    message(STATUS "Using ${CMAKE_CXX_STANDARD} as CMAKE_HIP_STANDARD")
    set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    enable_language(HIP)
  else()
    message(FATAL_ERROR "Requested device offload support, but no CUDA or HIP compilers available.")
  endif()
endif()

# --- PROJECT TARGETS --- #
add_subdirectory(include/layout_tiled)

if(LayoutTiled_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.16.0)
  # Prevent GTest from getting installed along with the library
  set(INSTALL_GTEST OFF)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests/layout_tiled)
endif()

if(LayoutTiled_BUILD_EXAMPLES)
  FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 11.1.4)
  FetchContent_MakeAvailable(fmt)
  add_subdirectory(examples)
endif()

if(LayoutTiled_BUILD_BENCHMARKS)
  # Fork adding support for tracking L1I/L1D/LLC events
  FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/dssgabriel/nanobench.git
    GIT_TAG origin/feature/cache-perf-events
    GIT_SHALLOW TRUE
  )
  FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 11.1.4)
  FetchContent_MakeAvailable(
    nanobench
    fmt
  )
  add_subdirectory(benchmarks)
endif()

# --- PACKAGING --- #
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/layout_tiled/layout-tiled-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Install CMake package config files
install(
  FILES
    ${PROJECT_BINARY_DIR}/cmake/layout_tiled/layout-tiled-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
)
install(
  EXPORT LayoutTiledTargets
  FILE layout-tiled-targets.cmake
  NAMESPACE layout_tiled::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
)
