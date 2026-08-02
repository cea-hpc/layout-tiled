set(
  ALL_BENCHMARKS
  gemm
  gemv
  stencil_2d
  stencil_3d
  tensor_contraction
)

if(LAYOUT_TILED_BENCH_ENABLE_OMP)
  find_package(OpenMP REQUIRED)
endif()
if(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  set(LAYOUT_TILED_BENCH_BLAS_LIBRARY "" CACHE FILEPATH "Explicit BLAS library used by the linear algebra benchmarks")
  set(LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR "" CACHE PATH "Directory containing cblas.h for the linear algebra benchmarks")
  if(LAYOUT_TILED_BENCH_BLAS_LIBRARY)
    add_library(layout_tiled.benchmark_blas UNKNOWN IMPORTED)
    set_target_properties(
      layout_tiled.benchmark_blas
      PROPERTIES IMPORTED_LOCATION "${LAYOUT_TILED_BENCH_BLAS_LIBRARY}"
    )
    find_package(Threads REQUIRED)
    set(LAYOUT_TILED_BENCH_BLAS_TARGET layout_tiled.benchmark_blas Threads::Threads)
  else()
    find_package(BLAS REQUIRED)
    set(LAYOUT_TILED_BENCH_BLAS_TARGET BLAS::BLAS)
  endif()
  if(LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR)
    set(LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR_RESOLVED "${LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR}")
  else()
    find_path(
      LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR_RESOLVED
      cblas.h
      /usr/include
      /usr/local/include
      /usr/include/openblas
    )
  endif()
  message("Benchmark BLAS library: ${LAYOUT_TILED_BENCH_BLAS_LIBRARY}")
  message("Benchmark BLAS include directory: ${LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR_RESOLVED}")
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
  if((${bench} STREQUAL "gemm" OR ${bench} STREQUAL "gemv") AND LAYOUT_TILED_BENCH_ENABLE_BLAS)
    message("Enabling BLAS for ${bench} benchmark")
    target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_ENABLE_BLAS)
    target_include_directories(
      layout_tiled.benchmarks.${bench}
      PRIVATE ${LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR_RESOLVED}
    )
    target_link_libraries(layout_tiled.benchmarks.${bench} PRIVATE ${LAYOUT_TILED_BENCH_BLAS_TARGET})
    if(
      LAYOUT_TILED_BENCH_BLAS_LIBRARY MATCHES "openblas"
      OR BLAS_openblas_LIBRARY
    )
      target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_OPENBLAS)
    endif()
  endif()
endforeach()
