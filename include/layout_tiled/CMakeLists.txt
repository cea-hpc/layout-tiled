# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

add_library(layout_tiled_compile_flags INTERFACE)
target_compile_options(
  layout_tiled_compile_flags
  INTERFACE
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
# Require C++23 w/o GNU extensions
target_compile_features(layout_tiled_compile_flags INTERFACE cxx_std_23)
set_target_properties(
  layout_tiled_compile_flags
  PROPERTIES
    CXX_EXTENSIONS
      OFF
)

add_library(layout_tiled INTERFACE)
add_library(layout_tiled::layout_tiled ALIAS layout_tiled)
target_sources(
  layout_tiled
  INTERFACE
    FILE_SET layout_tiled_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
      layout_right_tiled_right.hpp
      layout_right_tiled_left.hpp
      layout_left_tiled_right.hpp
      layout_left_tiled_left.hpp
)
target_sources(
  layout_tiled
  INTERFACE
    FILE_SET layout_tiled_details_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES details/common.hpp
)
target_link_libraries(layout_tiled INTERFACE layout_tiled_compile_flags)

# --- PACKAGING --- #
include(GNUInstallDirs)

# Export and install public headers
install(
  TARGETS
    layout_tiled
    layout_tiled_compile_flags
  EXPORT LayoutTiledTargets
  FILE_SET
  layout_tiled_public_headers
  FILE_SET
  layout_tiled_details_headers
)
