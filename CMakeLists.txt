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

cmake_minimum_required(VERSION 3.25)

project(layout-tiled LANGUAGES CXX VERSION 0.1.0 DESCRIPTION "Tiled layouts for C++23's std::mdspan")

# --- PROJECT-WIDE SETTINGS --- #
# Sensible defaults
set(CMAKE_COLOR_DIAGNOSTICS ON) # Enable colored compiler diagnostics for Ninja
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for the linter/static analyzer

# Require C++26 w/o GNU extensions
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- PROJECT OPTIONS --- #
option(LayoutTiled_BUILD_TESTS "Build tests" OFF)
option(LayoutTiled_BUILD_EXAMPLES "Build examples" OFF)
option(LayoutTiled_BUILD_BENCHMARKS "Build benchmarks" OFF)

# --- PROJECT TARGETS --- #
add_subdirectory(include/layout_tiled)

if(LayoutTiled_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG
      f8d7d77c06936315286eb55f8de22cd23c188571 # release-1.14.0
  )
  # Prevent GTest from getting installed along with the library
  set(INSTALL_GTEST OFF)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests/layout_tiled)
endif()

if(LayoutTiled_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(LayoutTiled_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/dssgabriel/nanobench.git
    GIT_TAG
      78c10d9effa994e05299eebd299a8736cd2eaee2 # adds support for L1D/LLC cache events
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nanobench)

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
  FILES ${PROJECT_BINARY_DIR}/cmake/layout_tiled/layout-tiled-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
)
install(
  EXPORT LayoutTiledTargets
  FILE layout-tiled-targets.cmake
  NAMESPACE layout_tiled::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
)
