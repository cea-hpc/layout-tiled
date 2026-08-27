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
  set(
    LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR
    ""
    CACHE PATH
    "Directory containing cblas.h for the linear algebra benchmarks"
  )
  if(LAYOUT_TILED_BENCH_BLAS_LIBRARY)
    add_library(layout_tiled.benchmark_blas UNKNOWN IMPORTED)
    set_target_properties(
      layout_tiled.benchmark_blas
      PROPERTIES IMPORTED_LOCATION "${LAYOUT_TILED_BENCH_BLAS_LIBRARY}"
    )
    find_package(Threads REQUIRED)
    set(LAYOUT_TILED_BENCH_BLAS_TARGET layout_tiled.benchmark_blas Threads::Threads)
  else()
    if(LAYOUT_TILED_BENCH_ENABLE_OMP)
      # FindBLAS may select a sequential OpenBLAS installation when several
      # threading variants are available. Prefer the explicitly named OpenMP
      # and pthread variants used by common OpenBLAS distributions.
      find_library(
        LAYOUT_TILED_BENCH_THREADED_BLAS_LIBRARY
        NAMES openblaso openblas_openmp openblasp openblas_pthread
        DOC "Automatically detected OpenMP- or pthread-enabled benchmark BLAS library"
      )
    endif()

    if(LAYOUT_TILED_BENCH_ENABLE_OMP AND LAYOUT_TILED_BENCH_THREADED_BLAS_LIBRARY)
      add_library(layout_tiled.benchmark_blas UNKNOWN IMPORTED)
      set_target_properties(
        layout_tiled.benchmark_blas
        PROPERTIES IMPORTED_LOCATION "${LAYOUT_TILED_BENCH_THREADED_BLAS_LIBRARY}"
      )
      find_package(Threads REQUIRED)
      set(LAYOUT_TILED_BENCH_BLAS_TARGET layout_tiled.benchmark_blas Threads::Threads)
    else()
      find_package(BLAS REQUIRED)
      set(LAYOUT_TILED_BENCH_BLAS_TARGET BLAS::BLAS)
    endif()
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
  if(LAYOUT_TILED_BENCH_BLAS_LIBRARY)
    set(LAYOUT_TILED_BENCH_BLAS_LIBRARY_RESOLVED "${LAYOUT_TILED_BENCH_BLAS_LIBRARY}")
  elseif(LAYOUT_TILED_BENCH_ENABLE_OMP AND LAYOUT_TILED_BENCH_THREADED_BLAS_LIBRARY)
    set(LAYOUT_TILED_BENCH_BLAS_LIBRARY_RESOLVED "${LAYOUT_TILED_BENCH_THREADED_BLAS_LIBRARY}")
  else()
    set(LAYOUT_TILED_BENCH_BLAS_LIBRARY_RESOLVED "${BLAS_LIBRARIES}")
  endif()
  message("Benchmark BLAS library: ${LAYOUT_TILED_BENCH_BLAS_LIBRARY_RESOLVED}")
  message("Benchmark BLAS include directory: ${LAYOUT_TILED_BENCH_BLAS_INCLUDE_DIR_RESOLVED}")
endif()

foreach(bench ${ALL_BENCHMARKS})
  add_executable(layout_tiled.benchmarks.${bench})
  target_sources(layout_tiled.benchmarks.${bench} PRIVATE ${bench}.cpp)
  if(bench STREQUAL "gemm")
    set(LAYOUT_TILED_BENCH_GEMM_TILE_I 64 CACHE STRING "GEMM tiled-layout extent on the I axis")
    set(LAYOUT_TILED_BENCH_GEMM_TILE_J 32 CACHE STRING "GEMM tiled-layout extent on the J axis")
    set(LAYOUT_TILED_BENCH_GEMM_TILE_K 64 CACHE STRING "GEMM tiled-layout extent on the K axis")
    target_compile_definitions(
      layout_tiled.benchmarks.${bench}
      PRIVATE
        LAYOUT_TILED_BENCH_GEMM_TILE_I=${LAYOUT_TILED_BENCH_GEMM_TILE_I}
        LAYOUT_TILED_BENCH_GEMM_TILE_J=${LAYOUT_TILED_BENCH_GEMM_TILE_J}
        LAYOUT_TILED_BENCH_GEMM_TILE_K=${LAYOUT_TILED_BENCH_GEMM_TILE_K}
    )
  endif()
  target_compile_options(
    layout_tiled.benchmarks.${bench}
    PRIVATE -Wall -Wextra -Wpedantic -pedantic -march=native -g3
  )
  target_compile_features(layout_tiled.benchmarks.${bench} PRIVATE cxx_std_23)
  set_target_properties(layout_tiled.benchmarks.${bench} PROPERTIES CXX_EXTENSIONS OFF)
  if(bench STREQUAL "gemm" OR bench STREQUAL "gemv" OR bench STREQUAL "tensor_contraction")
    set(benchmark_library_target iai::iai)
  else()
    set(benchmark_library_target layout_tiled::layout_tiled)
  endif()
  target_link_libraries(
    layout_tiled.benchmarks.${bench}
    PRIVATE ${benchmark_library_target} mdspan::mdspan fmt::fmt nanobench
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
    if(LAYOUT_TILED_BENCH_BLAS_LIBRARY_RESOLVED MATCHES "openblas" OR BLAS_openblas_LIBRARY)
      target_compile_definitions(layout_tiled.benchmarks.${bench} PRIVATE -DLAYOUT_TILED_BENCH_OPENBLAS)
    endif()
  endif()
endforeach()
