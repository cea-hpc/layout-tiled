find_package(mdspan)
find_package(GTest REQUIRED)

include(GoogleTest)

set(
  ALL_TESTS
  layout_left_tiled_left
  layout_left_tiled_right
  layout_right_tiled_left
  layout_right_tiled_right
)

foreach(test ${ALL_TESTS})
  add_executable(layout_tiled.tests.${test})
  target_sources(layout_tiled.tests.${test} PRIVATE ${test}.test.cpp)
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
