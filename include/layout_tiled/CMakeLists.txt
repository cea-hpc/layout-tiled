find_package(mdspan)

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
if(LayoutTiled_INSTALL_PACKAGE)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/layout-tiled-config.cmake.in"
    "${PROJECT_BINARY_DIR}/cmake/layout-tiled-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled"
    PATH_VARS
      PROJECT_NAME
      PROJECT_VERSION
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/cmake/layout-tiled-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
  )

  # Install CMake package config files
  install(
    FILES
      ${PROJECT_BINARY_DIR}/cmake/layout-tiled-config.cmake
      ${PROJECT_BINARY_DIR}/cmake/layout-tiled-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
    COMPONENT layout-tiled
  )
  install(
    EXPORT layout-tiled-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
    NAMESPACE layout_tiled::
    FILE layout-tiled-targets.cmake
    COMPONENT layout-tiled
  )

  # Install library headers
  install(
    TARGETS
      layout_tiled
    EXPORT layout-tiled-targets
    FILE_SET
    layout_tiled_public_headers
    FILE_SET
    layout_tiled_detail_headers
  )
endif()
