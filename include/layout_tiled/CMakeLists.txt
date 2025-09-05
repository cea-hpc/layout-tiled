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
    FILES layout_rtr.hpp layout_rtl.hpp layout_ltr.hpp layout_ltl.hpp
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
  target_compile_definitions(layout_tiled INTERFACE -DLAYOUT_TILED_ENABLE_DEVICE_OFFLOAD)
endif()

# --- PACKAGING --- #
include(GNUInstallDirs)
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
  EXPORT layout-tiled-targets
  FILE layout-tiled-targets.cmake
  NAMESPACE layout_tiled::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/layout_tiled
)

# Export and install library headers
install(
  TARGETS
    layout_tiled
    layout_tiled.compile_flags
  EXPORT layout-tiled-targets
  FILE_SET
  layout_tiled_public_headers
  FILE_SET
  layout_tiled_detail_headers
)
