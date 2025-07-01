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
  target_compile_options(
    layout_tiled.examples.${example}
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
  target_compile_features(layout_tiled.examples.${example} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.examples.${example}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.examples.${example}
    PRIVATE
      layout_tiled::layout_tiled
      mdspan::mdspan
      fmt::fmt
  )
endforeach()
