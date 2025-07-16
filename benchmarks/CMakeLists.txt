find_package(mdspan)
find_package(fmt)
find_package(nanobench)

set(
  ALL_BENCHMARKS
  gemm
  stencil_2d
  stencil_3d
)

option(
  LayoutTiled_BENCHMARKS_ENABLE_CHECKS
  "Enable checking the computed results of benchmarks. Default: ON. Values: { ON, OFF }."
  ON
)
option(
  LayoutTiled_BENCHMARKS_ENABLE_OPENMP
  "Enable building benchmarks with OpenMP support. Default: ON. Values: { ON, OFF }."
  ON
)
option(
  LayoutTiled_BENCHMARKS_ENABLE_BLAS
  "Enable building the GEMM benchmark with a BLAS version. Default: ON. Values: { ON, OFF }."
  ON
)

if(LayoutTiled_BENCHMARKS_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
if(LayoutTiled_BENCHMARKS_ENABLE_BLAS)
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
  target_compile_options(layout_tiled.benchmarks.${bench} PRIVATE -march=native)
  target_link_libraries(
    layout_tiled.benchmarks.${bench}
    PRIVATE
      layout_tiled::layout_tiled
      mdspan::mdspan
      fmt::fmt
      nanobench
  )
  if(LayoutTiled_BENCHMARKS_ENABLE_OPENMP)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_ENABLE_OPENMP)
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(LayoutTiled_BENCHMARKS_ENABLE_CHECKS)
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_ENABLE_CHECKS)
  endif()
  if(${bench} STREQUAL "gemm" AND LayoutTiled_BENCHMARKS_ENABLE_BLAS)
    message("Enabling BLAS for DGEMM benchmark")
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_ENABLE_BLAS)
    target_include_directories(layout_tiled.benchmarks.${bench} PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE BLAS::BLAS)
  endif()
endforeach()
