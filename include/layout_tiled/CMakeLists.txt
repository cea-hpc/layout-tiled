
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
target_link_libraries(layout_tiled INTERFACE mdspan::mdspan)

# --- PACKAGING --- #
include(GNUInstallDirs)

# Export and install public headers
install(
  TARGETS
    layout_tiled
  EXPORT LayoutTiledTargets
  FILE_SET
  layout_tiled_public_headers
  FILE_SET
  layout_tiled_detail_headers
)
