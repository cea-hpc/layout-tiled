add_library(layout_tiled.compile_flags INTERFACE)
target_compile_options(
  layout_tiled.compile_flags
  INTERFACE
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
target_compile_features(layout_tiled.compile_flags INTERFACE cxx_std_23)
set_target_properties(
  layout_tiled.compile_flags
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
    FILE_SET layout_tiled_detail_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES detail/common.hpp
)
target_link_libraries(
  layout_tiled
  INTERFACE
    layout_tiled.compile_flags
    mdspan::mdspan
)

if(LayoutTiled_ENABLE_DEVICE_OFFLOAD)
  target_compile_definitions(layout_tiled INTERFACE -DLAYOUT_TILED_ENABLE)
endif()

# --- PACKAGING --- #
include(GNUInstallDirs)

# Export and install public headers
install(
  TARGETS
    layout_tiled
    layout_tiled.compile_flags
  EXPORT LayoutTiledTargets
  FILE_SET
  layout_tiled_public_headers
  FILE_SET
  layout_tiled_detail_headers
)
