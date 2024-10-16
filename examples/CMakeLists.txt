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

cmake_minimum_required(VERSION 3.23)

project(layout-tiled-examples LANGUAGES CXX DESCRIPTION "Examples for layout-tiled")

find_package(fmt REQUIRED)

if(NOT TARGET layout_tiled::layout_tiled)
  find_package(layout-tiled REQUIRED)
endif()

add_executable(simple)
target_sources(simple PRIVATE simple_test.cpp)
target_compile_options(
  simple
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
target_compile_features(simple PRIVATE cxx_std_23)
set_target_properties(
  simple
  PROPERTIES
    CXX_EXTENSIONS
      OFF
)
target_link_libraries(
  simple
  PRIVATE
    layout_tiled::layout_tiled
    fmt::fmt
)
