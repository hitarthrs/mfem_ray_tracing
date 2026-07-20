cmake_minimum_required(VERSION 3.16)
project(mfem_raytracing CXX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(MFEM_DIR "$ENV{HOME}/mesh_software/mfem-4.9/build" CACHE PATH "MFEM build directory")
list(APPEND CMAKE_PREFIX_PATH "${MFEM_DIR}")
find_package(mfem REQUIRED)

option(MFEM_RAYTRACING_ENABLE_EMBREE
       "Build Embree ray tracing library (src/embree)" OFF)
set(EMBREE_DIR "" CACHE PATH "Embree install prefix (optional)")

# Core library: degree reduction → watertight bilinear leaves (+ mesh helpers).
add_library(mfem_raytracing
    src/cartesian_mesh_builder.cpp
    src/cartesian_mesh_naming.cpp
    src/nurbs_mesh_operations.cpp
    src/bezier_degree_reduction.cpp
    src/b_spline_curve_reduction.cpp
    src/nurbs_degree_reduction.cpp
    src/curve_reduction_domain.cpp
    src/curve_reduction_error_budget.cpp
    src/curve_degree_reduction.cpp
    src/surface_reduction_domain.cpp
    src/surface_degree_reduction.cpp
    src/surface_multistep_reduction.cpp
    src/bilinear_leaf_extraction.cpp
    src/leaf_patch_loader.cpp
)
target_include_directories(mfem_raytracing PUBLIC include)
target_link_libraries(mfem_raytracing PUBLIC mfem)

# ------------------------------------------------------------------------------
# Embree ray tracing (CPU BVH); see xdg/CMakeLists.txt for the same pattern.
# ------------------------------------------------------------------------------
if(MFEM_RAYTRACING_ENABLE_EMBREE)
  include(Embree)
  mfem_raytracing_find_embree()

  file(GLOB MFEM_RAYTRACING_EMBREE_EXTRA_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/embree/*.cpp")
  add_library(mfem_raytracing_embree
      src/bilinear_intersect.cpp
      ${MFEM_RAYTRACING_EMBREE_EXTRA_SOURCES}
  )
  target_include_directories(mfem_raytracing_embree
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  mfem_raytracing_link_embree(mfem_raytracing_embree)
  message(STATUS "mfem_raytracing_embree sources: src/bilinear_intersect.cpp ${MFEM_RAYTRACING_EMBREE_EXTRA_SOURCES}")
endif()

add_executable(make_cart_mesh src/make_cart_mesh.cpp)
target_link_libraries(make_cart_mesh PRIVATE mfem_raytracing)

add_executable(bench_curve_degree_reduction tests/bench_curve_degree_reduction.cpp)
target_link_libraries(bench_curve_degree_reduction PRIVATE mfem_raytracing)

add_executable(bench_surface_degree_reduction tests/bench_surface_degree_reduction.cpp)
target_link_libraries(bench_surface_degree_reduction PRIVATE mfem_raytracing)

# Multi-step reduction -> bilinear leaf AABB JSON (demo_leaf_bboxes.py port).
add_executable(export_leaf_bboxes src/export_leaf_bboxes.cpp)
target_include_directories(export_leaf_bboxes PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(export_leaf_bboxes PRIVATE mfem_raytracing)

# Watertight (conforming + coalesce) bilinear patch export for any (p, q)
# surface; input surfaces from python_experiments/export_surface_inputs.py.
add_executable(export_bilinear_patches src/export_bilinear_patches.cpp)
target_link_libraries(export_bilinear_patches PRIVATE mfem_raytracing)

enable_testing()

add_executable(run_tests
    tests/main.cpp
    tests/test_cartesian_mesh.cpp
    tests/test_nurbs_mesh.cpp
    tests/test_b_spline_curve_reduction.cpp
    tests/test_bezier_degree_reduction.cpp
    tests/test_nurbs_degree_reduction.cpp
    tests/test_curve_degree_reduction.cpp
    tests/test_surface_degree_reduction.cpp
    tests/test_surface_multistep_reduction.cpp
)
target_link_libraries(run_tests PRIVATE mfem_raytracing)

if(MFEM_RAYTRACING_ENABLE_EMBREE)
  target_sources(run_tests PRIVATE
      tests/test_bilinear_intersect.cpp
      tests/test_embree_raytracer.cpp
  )
  target_link_libraries(run_tests PRIVATE mfem_raytracing_embree)

  add_executable(render_leaf_patches src/render_leaf_patches.cpp)
  target_link_libraries(render_leaf_patches PRIVATE mfem_raytracing_embree mfem_raytracing)

  add_executable(export_scene_json src/export_scene_json.cpp)
  target_link_libraries(export_scene_json PRIVATE mfem_raytracing_embree mfem_raytracing)
endif()

add_test(NAME unit_tests COMMAND run_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
