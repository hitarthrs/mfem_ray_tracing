# Embree discovery and target wiring (pattern from xdg/CMakeLists.txt).

set(MFEM_RAYTRACING_EMBREE_FOUND FALSE)

function(mfem_raytracing_find_embree)
  if(MFEM_RAYTRACING_EMBREE_FOUND)
    return()
  endif()

  if(EMBREE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${EMBREE_DIR}")
  endif()

  # Embree's config does not support true version ranges; try v4 then v3.
  find_package(embree 4.0.0...<4.1.0 QUIET PATHS ${CMAKE_PREFIX_PATH})
  if(NOT DEFINED EMBREE_VERSION)
    message(WARNING
      "Could not find Embree 4.x; searching for Embree 3.x...")
    find_package(embree 3.0.0...<4.0.0 REQUIRED PATHS ${CMAKE_PREFIX_PATH})
  endif()

  if(NOT DEFINED EMBREE_VERSION)
    message(FATAL_ERROR "Embree package was not found. Set -DEMBREE_DIR=/path/to/embree")
  endif()

  if(EMBREE_VERSION VERSION_LESS "3.6.1")
    message(FATAL_ERROR "mfem_raytracing requires Embree v3.6.1 or higher.")
  endif()

  set(MFEM_RAYTRACING_EMBREE_FOUND TRUE PARENT_SCOPE)
  set(MFEM_RAYTRACING_EMBREE_VERSION "${EMBREE_VERSION}" PARENT_SCOPE)
  message(STATUS "Found Embree ${EMBREE_VERSION}")
endfunction()

# Link the CMake ``embree`` imported target and set version preprocessor macros.
function(mfem_raytracing_link_embree target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "mfem_raytracing_link_embree: target '${target}' does not exist")
  endif()
  if(NOT TARGET embree)
    message(FATAL_ERROR "mfem_raytracing_link_embree: Embree target not available")
  endif()

  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    set(_scope INTERFACE)
  else()
    set(_scope PUBLIC)
  endif()

  target_link_libraries(${target} ${_scope} embree)
  target_compile_definitions(${target} ${_scope} MFEM_RAYTRACING_ENABLE_EMBREE)

  if(MFEM_RAYTRACING_EMBREE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    target_compile_definitions(${target} ${_scope} MFEM_RAYTRACING_EMBREE4)
  else()
    target_compile_definitions(${target} ${_scope} MFEM_RAYTRACING_EMBREE3)
  endif()
endfunction()
