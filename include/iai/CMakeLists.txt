add_library(iai INTERFACE)
add_library(iai::iai ALIAS iai)
target_sources(
  iai
  INTERFACE
    FILE_SET iai_public_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
    FILES bindings.hpp md_reduce.hpp
    FILE_SET iai_detail_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
    FILES
      detail/access_transform.hpp
      detail/axis.hpp
      detail/projected_view.hpp
      detail/md_reduce.hpp
      detail/layout_tiled_mapping.hpp
      detail/native_tiled_blocked_fragment_reduce.hpp
      detail/native_tiled_cache_block.hpp
      detail/native_tiled_fragment_reduce.hpp
      detail/native_tiled_reduce.hpp
      detail/native_tiled_reduce_policy.hpp
      detail/native_tiled_view.hpp
)
target_link_libraries(iai INTERFACE layout_tiled::layout_tiled)

include(GNUInstallDirs)

install(
  EXPORT iai-targets
  FILE iai-targets.cmake
  NAMESPACE iai::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout-tiled
)
install(
  TARGETS iai
  EXPORT iai-targets
  FILE_SET iai_public_headers
  FILE_SET iai_detail_headers
)
