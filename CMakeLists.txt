cmake_minimum_required(VERSION 3.28)

project(layout-tiled VERSION 0.4.0 LANGUAGES CXX DESCRIPTION "Tiled layouts for `std::mdspan`")

# --- PROJECT OPTIONS --- #
option(LAYOUT_TILED_BUILD_TESTS "Enable building tests. Default: OFF. Values: { ON, OFF}." OFF)
option(LAYOUT_TILED_BUILD_EXAMPLES "Enable building examples. Default: OFF. Values: { ON, OFF }." OFF)
option(LAYOUT_TILED_BUILD_BENCHMARKS "Enable building benchmarks. Default: OFF. Values: { ON, OFF }." OFF)
option(LAYOUT_TILED_ENABLE_KOKKOS_MDSPAN "Build with Kokkos reference mdspan implementation. Default: ON. Values: { ON, OFF }." ON)
option(LAYOUT_TILED_ENABLE_DEVICE_OFFLOAD "Enable device offload. Default: OFF. Values: { ON, OFF }." OFF)
option(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN "Enable submdspan support (submdspan_mapping). Default: OFF. Values: { ON, OFF }." OFF)

# Benchmark options
option(LAYOUT_TILED_BENCH_ENABLE_CHECKS "Enable checking benchmarks results. Default: ON. Values: { ON, OFF }." OFF)
option(LAYOUT_TILED_BENCH_ENABLE_OMP "Enable building benchmarks with OpenMP. Default: OFF. Values: { ON, OFF }." OFF)
option(LAYOUT_TILED_BENCH_ENABLE_BLAS "Enable building GEMM benchmark with BLAS. Default: ON. Values: { ON, OFF }." OFF)

# Resolve device offload
if(LAYOUT_TILED_ENABLE_DEVICE_OFFLOAD)
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

# --- PROJECT DEPENDENCIES --- #
include(FetchContent)
if(LAYOUT_TILED_ENABLE_KOKKOS_MDSPAN)
  FetchContent_Declare(mdspan GIT_REPOSITORY https://github.com/kokkos/mdspan.git GIT_TAG origin/stable)
  FetchContent_MakeAvailable(mdspan)
else()
  include(CheckIncludeFileCXX)
  check_include_file_cxx(mdspan LAYOUT_TILED_HAVE_STD_MDSPAN)
  if(NOT LAYOUT_TILED_HAVE_STD_MDSPAN)
    message(FATAL_ERROR
      "Standard library does not provide <mdspan>. "
      "Set -DLAYOUT_TILED_ENABLE_KOKKOS_MDSPAN=ON "
      "or use a C++23 compiler with mdspan support."
    )
  endif()
endif()

# Resolve submdspan support
if(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)
  if(LAYOUT_TILED_ENABLE_KOKKOS_MDSPAN)
    message(STATUS "submdspan support enabled (provided by Kokkos mdspan)")
  else()
    include(CheckCXXSourceCompiles)
    set(layout_tiled_saved_cxx_standard "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 23)
    check_cxx_source_compiles("
#include <mdspan>
int main() {
  using ext_t = std::dextents<size_t, 2>;
  ext_t e{4, 5};
  auto sub_exts = std::submdspan_extents(e, std::integral_constant<size_t, 0>{}, std::integral_constant<size_t, 1>{});
  using result_t = std::submdspan_mapping_result<decltype(sub_exts)>;
  result_t r{};
  (void)sub_exts;
  (void)r;
  return 0;
}
" LAYOUT_TILED_HAVE_SUBMDSPAN)
    if(layout_tiled_saved_cxx_standard)
      set(CMAKE_CXX_STANDARD "${layout_tiled_saved_cxx_standard}")
    else()
      unset(CMAKE_CXX_STANDARD)
    endif()
    if(NOT LAYOUT_TILED_HAVE_SUBMDSPAN)
      message(FATAL_ERROR
        "Requested submdspan support (-DLAYOUT_TILED_ENABLE_P2630_SUBMDSPAN=ON), "
        "but the standard library's <mdspan> lacks submdspan features "
        "(P2630 submdspan_mapping_result / submdspan_extents). "
        "Use a newer standard library, or set -DLAYOUT_TILED_ENABLE_KOKKOS_MDSPAN=ON."
      )
    endif()
  endif()
endif()

# --- PROJECT TARGETS --- #
add_subdirectory(include/layout_tiled)

if(LAYOUT_TILED_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.16.0)
  # Prevent GTest from getting installed along with the library
  set(INSTALL_GTEST OFF)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests/layout_tiled)
endif()

if(LAYOUT_TILED_BUILD_EXAMPLES)
  FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 11.1.4)
  FetchContent_MakeAvailable(fmt)
  add_subdirectory(examples)
endif()

if(LAYOUT_TILED_BUILD_BENCHMARKS)
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
