include(GoogleTest)

set(
  ALL_TESTS
  layout_ltl
  layout_ltr
  layout_rtl
  layout_rtr
  iai
  iai_reduce
)

foreach(test ${ALL_TESTS})
  add_executable(layout_tiled.tests.${test})
  target_sources(layout_tiled.tests.${test} PRIVATE ${test}.test.cpp)
  target_compile_options(
    layout_tiled.tests.${test}
    PRIVATE -Wall -Wextra -Wshadow -Wpedantic -Wcast-align -Wconversion -Wmissing-include-dirs -Wformat=2
  )
  target_compile_features(layout_tiled.tests.${test} PRIVATE cxx_std_23)
  set_target_properties(layout_tiled.tests.${test} PROPERTIES CXX_EXTENSIONS OFF)
  if(test MATCHES "^iai")
    set(library_target iai::iai)
  else()
    set(library_target layout_tiled::layout_tiled)
  endif()
  target_link_libraries(
    layout_tiled.tests.${test}
    PRIVATE ${library_target} mdspan::mdspan GTest::gtest GTest::gtest_main
  )

  gtest_discover_tests(layout_tiled.tests.${test})
endforeach()
