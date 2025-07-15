cmake_minimum_required(VERSION 3.25)

include(FetchContent)

if(NOT LAYOUT_TILED_LOCKFILE)
  set(LAYOUT_TILED_LOCKFILE "lockfile.json" CACHE FILEPATH "Path to the dependency lockfile for layout-tiled.")
endif()

set(LayoutTiled_projectDir "${CMAKE_CURRENT_LIST_DIR}/..")
message(TRACE "LayoutTiled_projectDir=\"${LayoutTiled_projectDir}\"")

message(TRACE "LAYOUT_TILED_LOCKFILE=\"${LAYOUT_TILED_LOCKFILE}\"")
file(REAL_PATH "${LAYOUT_TILED_LOCKFILE}" LayoutTiled_lockfile BASE_DIRECTORY "${LayoutTiled_projectDir}" EXPAND_TILDE)
message(DEBUG "Using lockfile: \"${LayoutTiled_lockfile}\"")

# Force CMake to reconfigure the project if the lockfile changes
set_property(
  DIRECTORY
    "${LayoutTiled_projectDir}"
  APPEND
  PROPERTY
    CMAKE_CONFIGURE_DEPENDS
      "${LayoutTiled_lockfile}"
)

# For more on the protocol for this function, see:
# https://cmake.org/cmake/help/latest/command/cmake_language.html#provider-commands
function(LayoutTiled_provideDependency method package_name)
  # Read the lockfile
  file(READ "${LayoutTiled_lockfile}" LayoutTiled_rootObj)

  # Get the "dependencies" field and store it in LayoutTiled_dependenciesObj
  string(JSON LayoutTiled_dependenciesObj ERROR_VARIABLE LayoutTiled_error GET "${LayoutTiled_rootObj}" "dependencies")
  if(LayoutTiled_error)
    message(FATAL_ERROR "${LayoutTiled_lockfile}: ${LayoutTiled_error}")
  endif()

  # Get the length of the libraries array and store it in LayoutTiled_dependenciesObj
  string(JSON LayoutTiled_numDependencies ERROR_VARIABLE LayoutTiled_error LENGTH "${LayoutTiled_dependenciesObj}")
  if(LayoutTiled_error)
    message(FATAL_ERROR "${LayoutTiled_lockfile}: ${LayoutTiled_error}")
  endif()

  if(LayoutTiled_numDependencies EQUAL 0)
    return()
  endif()

  # Loop over each dependency object
  math(EXPR LayoutTiled_maxIndex "${LayoutTiled_numDependencies} - 1")
  foreach(LayoutTiled_index RANGE "${LayoutTiled_maxIndex}")
    set(LayoutTiled_errorPrefix "${LayoutTiled_lockfile}, dependency ${LayoutTiled_index}")

    # Get the dependency object at LayoutTiled_index
    # and store it in LayoutTiled_depObj
    string(
      JSON
      LayoutTiled_depObj
      ERROR_VARIABLE LayoutTiled_error
      GET "${LayoutTiled_dependenciesObj}"
      "${LayoutTiled_index}"
    )
    if(LayoutTiled_error)
      message(FATAL_ERROR "${LayoutTiled_errorPrefix}: ${LayoutTiled_error}")
    endif()

    # Get the "name" field and store it in LayoutTiled_name
    string(JSON LayoutTiled_name ERROR_VARIABLE LayoutTiled_error GET "${LayoutTiled_depObj}" "name")
    if(LayoutTiled_error)
      message(FATAL_ERROR "${LayoutTiled_errorPrefix}: ${LayoutTiled_error}")
    endif()

    # Get the "package_name" field and store it in LayoutTiled_pkgName
    string(JSON LayoutTiled_pkgName ERROR_VARIABLE LayoutTiled_error GET "${LayoutTiled_depObj}" "package_name")
    if(LayoutTiled_error)
      message(FATAL_ERROR "${LayoutTiled_errorPrefix}: ${LayoutTiled_error}")
    endif()

    # Get the "git_repository" field and store it in LayoutTiled_repo
    string(JSON LayoutTiled_repo ERROR_VARIABLE LayoutTiled_error GET "${LayoutTiled_depObj}" "git_repository")
    if(LayoutTiled_error)
      message(FATAL_ERROR "${LayoutTiled_errorPrefix}: ${LayoutTiled_error}")
    endif()

    # Get the "git_tag" field and store it in LayoutTiled_tag
    string(JSON LayoutTiled_tag ERROR_VARIABLE LayoutTiled_error GET "${LayoutTiled_depObj}" "git_tag")
    if(LayoutTiled_error)
      message(FATAL_ERROR "${LayoutTiled_errorPrefix}: ${LayoutTiled_error}")
    endif()

    if(method STREQUAL "FIND_PACKAGE")
      if(package_name STREQUAL LayoutTiled_pkgName)
        string(
          APPEND
          LayoutTiled_debug
          "Redirecting find_package calls for ${LayoutTiled_pkgName} "
          "to FetchContent logic fetching ${LayoutTiled_repo} at "
          "${LayoutTiled_tag} according to ${LayoutTiled_lockfile}."
        )
        message(DEBUG "${LayoutTiled_debug}")
        FetchContent_Declare(
          "${LayoutTiled_name}"
          GIT_REPOSITORY "${LayoutTiled_repo}"
          GIT_TAG "${LayoutTiled_tag}"
          EXCLUDE_FROM_ALL
        )
        set(INSTALL_GTEST OFF) # Disable GoogleTest installation
        FetchContent_MakeAvailable("${LayoutTiled_name}")

        # Important! <PackageName>_FOUND tells CMake that `find_package` is
        # not needed for this package anymore
        set("${LayoutTiled_pkgName}_FOUND" TRUE PARENT_SCOPE)
      endif()
    endif()
  endforeach()
endfunction()

cmake_language(SET_DEPENDENCY_PROVIDER LayoutTiled_provideDependency SUPPORTED_METHODS FIND_PACKAGE)
