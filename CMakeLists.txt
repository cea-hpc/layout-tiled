cmake_minimum_required(VERSION 3.25)

project(layout-tiled LANGUAGES CXX DESCRIPTION "Tiled layouts for std::mdspan" VERSION 0.4.0)

# --- PROJECT OPTIONS --- #
option(
  LayoutTiled_BUILD_TESTS
  "Enable building tests and test infrastructure. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_BUILD_EXAMPLES
  "Enable building examples. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_BUILD_BENCHMARKS
  "Enable building benchmarks. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  LayoutTiled_INSTALL_PACKAGE
  "Enable installing library as CMake package. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)

# --- PROJECT DEPENDENCIES --- #
include(CTest)

# --- PROJECT TARGETS --- #
add_subdirectory(include/layout_tiled)

if(LayoutTiled_BUILD_TESTS)
  add_subdirectory(tests/layout_tiled)
endif()

if(LayoutTiled_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(LayoutTiled_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# --- PACKAGING --- #
if(LayoutTiled_INSTALL_PACKAGE)
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
endif()
