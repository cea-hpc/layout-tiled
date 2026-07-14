include(GoogleTest)

set(static_invariant_include_dirs "${PROJECT_SOURCE_DIR}/include;${mdspan_SOURCE_DIR}/include")
set(static_invariant_compile_definitions)
if(LAYOUT_TILED_ENABLE_DEVICE_OFFLOAD)
  list(APPEND static_invariant_compile_definitions -DLAYOUT_TILED_ENABLE_DEVICE_OFFLOAD)
endif()

function(expect_static_invariant_failure layout invariant diagnostic)
  try_compile(
    compile_succeeded
    SOURCE_FROM_FILE
      "static_invariant_${layout}_${invariant}.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/static_invariant.cpp"
    COMPILE_DEFINITIONS
      "-DLAYOUT_TILED_TEST_LAYOUT=${layout}"
      "-DLAYOUT_TILED_TEST_INVARIANT=${invariant}"
      ${static_invariant_compile_definitions}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${static_invariant_include_dirs}"
    CXX_STANDARD
      23
    CXX_STANDARD_REQUIRED
      ON
    CXX_EXTENSIONS
      OFF
    NO_CACHE
    OUTPUT_VARIABLE
      compiler_output
  )
  if(compile_succeeded)
    message(FATAL_ERROR "Static invariant test ${layout}/${invariant} unexpectedly compiled")
  endif()
  if(NOT compiler_output MATCHES "${diagnostic}")
    message(FATAL_ERROR
      "Static invariant test ${layout}/${invariant} failed for an unexpected reason:\n${compiler_output}"
    )
  endif()
endfunction()

foreach(layout RANGE 0 3)
  expect_static_invariant_failure(${layout} 0 "all static extents must be multiples")
  expect_static_invariant_failure(${layout} 1 "product of static extents must be representable")
  expect_static_invariant_failure(${layout} 2 "tile size must be representable")
endforeach()

set(
  ALL_TESTS
  common
  layout_ltl
  layout_ltr
  layout_rtl
  layout_rtr
)

foreach(test ${ALL_TESTS})
  add_executable(layout_tiled.tests.${test})
  target_sources(layout_tiled.tests.${test} PRIVATE ${test}.test.cpp)
  target_compile_options(
    layout_tiled.tests.${test}
    PRIVATE
      -Wall
      -Wextra
      -Wshadow
      -Wpedantic
      -Wcast-align
      -Wconversion
      -Wmissing-include-dirs
      -Wformat=2
  )
  target_compile_features(layout_tiled.tests.${test} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.tests.${test}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.tests.${test}
    PRIVATE
      layout_tiled::layout_tiled
      mdspan::mdspan
      GTest::gtest
      GTest::gtest_main
  )

  gtest_discover_tests(layout_tiled.tests.${test})
endforeach()
