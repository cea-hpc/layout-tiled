set(
  ALL_BENCHMARKS
  gemm
  stencil_2d
  stencil_3d
  tensor_contraction
)

if(LAYOUT_TILED_BENCH_ENABLE_OMP)
  find_package(OpenMP REQUIRED)
endif()
if(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  find_package(BLAS REQUIRED)
  find_path(
    BLAS_INCLUDE_DIRS
    cblas.h
    /usr/include
    /usr/local/include
    /usr/include/openblas
  )
  message("BLA_VENDOR: ${BLA_VENDOR}")
  message("BLAS_INCLUDE_DIRS: ${BLAS_INCLUDE_DIRS}")
endif()

foreach(bench ${ALL_BENCHMARKS})
  add_executable(layout_tiled.benchmarks.${bench})
  target_sources(layout_tiled.benchmarks.${bench} PRIVATE ${bench}.cpp)
  target_compile_options(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -pedantic
      -march=native
      -g3
  )
  target_compile_features(layout_tiled.benchmarks.${bench} PRIVATE cxx_std_23)
  set_target_properties(
    layout_tiled.benchmarks.${bench}
    PROPERTIES
      CXX_EXTENSIONS
        OFF
  )
  target_link_libraries(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      layout_tiled::layout_tiled
      mdspan::mdspan
      fmt::fmt
      nanobench
  )
  if(LAYOUT_TILED_BENCH_ENABLE_OMP)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_ENABLE_OMP)
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_ENABLE_CHECKS)
  endif()
  if(${bench} STREQUAL "gemm" AND LAYOUT_TILED_BENCH_ENABLE_BLAS)
    message("Enabling BLAS for GEMM benchmark")
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_ENABLE_BLAS)
    target_include_directories(layout_tiled.benchmarks.${bench} PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE BLAS::BLAS)
  endif()
endforeach()
