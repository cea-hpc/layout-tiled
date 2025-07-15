# This toolchain file is not meant to be used directly,
# but to be invoked by CMake preset and GitHub CI.
#
# This toolchain file configures for GNU family of compiler.
#
# LAYOUT_TILED_BUILDSYS_SANITIZER:
# This optional CMake parameter is not meant for public use and is subject to
# change.
# Possible values:
# - MaxSan: configures gcc and g++ to use all available non-conflicting
#           sanitizers.
# - TSan:   configures gcc and g++ to enable the use of thread sanitizer

include_guard(GLOBAL)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

if(LAYOUT_TILED_BUILDSYS_SANITIZER STREQUAL "MaxSan")
  set(
    SANITIZER_FLAGS
    "-fsanitize=address -fsanitize=leak -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize=undefined -fsanitize-undefined-trap-on-error"
  )
elseif(LAYOUT_TILED_BUILDSYS_SANITIZER STREQUAL "TSan")
  set(SANITIZER_FLAGS "-fsanitize=thread")
endif()

set(
  DEBUG_FLAGS
  "-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wdouble-promotion -Wformat=2 -fno-omit-frame-pointer -fno-inline -fno-common -fstack-protector-strong -ftrapv -pipe ${SANITIZER_FLAGS}"
)

set(CMAKE_C_FLAGS_DEBUG_INIT "${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "${DEBUG_FLAGS}")

set(RELEASE_FLAGS "-O3 ${SANITIZER_FLAGS}")

set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "${RELEASE_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE_INIT "${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "${RELEASE_FLAGS}")
