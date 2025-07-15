find_package(mdspan)
find_package(fmt)

set(
  ALL_EXAMPLES
  layout_right_tiled_right
  layout_right_tiled_left
  layout_left_tiled_right
  layout_left_tiled_left
)

foreach(example ${ALL_EXAMPLES})
  add_executable(layout_tiled.examples.${example})
  target_sources(layout_tiled.examples.${example} PRIVATE ${example}.cpp)
  target_link_libraries(
    layout_tiled.examples.${example}
    PRIVATE
      layout_tiled::layout_tiled
      mdspan::mdspan
      fmt::fmt
  )
endforeach()
