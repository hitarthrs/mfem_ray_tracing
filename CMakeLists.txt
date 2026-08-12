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
    src/mesh/cartesian_mesh_builder.cpp
    src/mesh/cartesian_mesh_naming.cpp
    src/mesh/nurbs_mesh_operations.cpp
    src/reduction/bezier_degree_reduction.cpp
    src/reduction/b_spline_curve_reduction.cpp
    src/reduction/nurbs_degree_reduction.cpp
    src/reduction/curve_reduction_domain.cpp
    src/reduction/curve_reduction_error_budget.cpp
    src/reduction/curve_degree_reduction.cpp
    src/reduction/surface_reduction_domain.cpp
    src/reduction/surface_degree_reduction.cpp
    src/reduction/surface_multistep_reduction.cpp
    src/reduction/bilinear_leaf_extraction.cpp
    src/reduction/hard_seam_bilinearization.cpp
    # Leaf JSON I/O is used by reduction/shell workflows without Embree.
    src/embree/leaf_patch_loader.cpp
    src/pipeline/reduce.cpp
    src/pipeline/multi_patch_tmesh.cpp
    src/tspline/tspline.cpp
    src/tspline/tspline_seam_types.cpp
    src/tspline/tspline_patch_interfaces.cpp
    src/tspline/tspline_leaf_assembly.cpp
    src/tspline/tspline_bilinear_ops.cpp
    src/tspline/tspline_average_merge.cpp
    src/tspline/tspline_strip_builder.cpp
    src/tspline/tspline_degree_one_bake.cpp
    src/tspline/tspline_error_accounting.cpp
    src/tspline/tspline_corner_collar.cpp
    src/tspline/tspline_shell_watertightness.cpp
    src/tspline/tspline_shell_composer.cpp
    src/tspline/tspline_shell_json.cpp
)
target_include_directories(mfem_raytracing PUBLIC include)
target_link_libraries(mfem_raytracing PUBLIC mfem)

# ------------------------------------------------------------------------------
# Embree ray tracing (CPU BVH); see xdg/CMakeLists.txt for the same pattern.
# ------------------------------------------------------------------------------
if(MFEM_RAYTRACING_ENABLE_EMBREE)
  include(Embree)
  mfem_raytracing_find_embree()

  add_library(mfem_raytracing_embree
      src/embree/bilinear_intersect.cpp
      src/embree/raytracer.cpp
  )
  target_include_directories(mfem_raytracing_embree
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  mfem_raytracing_link_embree(mfem_raytracing_embree)
  target_link_libraries(mfem_raytracing_embree PUBLIC mfem_raytracing)
  message(STATUS "mfem_raytracing_embree: bilinear_intersect + raytracer")
endif()

# ------------------------------------------------------------------------------
# Apps / tools (not part of the library)
# ------------------------------------------------------------------------------
add_executable(make_cart_mesh apps/make_cart_mesh.cpp)
target_link_libraries(make_cart_mesh PRIVATE mfem_raytracing)

add_executable(bench_curve_degree_reduction tests/bench_curve_degree_reduction.cpp)
target_link_libraries(bench_curve_degree_reduction PRIVATE mfem_raytracing)

add_executable(bench_surface_degree_reduction tests/bench_surface_degree_reduction.cpp)
target_link_libraries(bench_surface_degree_reduction PRIVATE mfem_raytracing)

# Multi-step reduction -> bilinear leaf AABB JSON (demo_leaf_bboxes.py port).
add_executable(export_leaf_bboxes apps/export_leaf_bboxes.cpp)
target_include_directories(export_leaf_bboxes PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(export_leaf_bboxes PRIVATE mfem_raytracing)

# Watertight (conforming + coalesce) bilinear patch export for any (p, q)
# surface; input surfaces from python_experiments/export_surface_inputs.py.
add_executable(export_bilinear_patches apps/export_bilinear_patches.cpp)
target_link_libraries(export_bilinear_patches PRIVATE mfem_raytracing)

# Full multi-span catalog patches -> independent hard-seam bilinear leaves.
add_executable(export_hard_seam_patches apps/export_hard_seam_patches.cpp)
target_link_libraries(export_hard_seam_patches PRIVATE mfem_raytracing)

# Full hard-seam leaf scene -> local degree-one T-spline strips -> baked
# bilinear runtime leaves.  Strict mode gates JSON emission on the combined
# source-ownership and geometric watertightness certificate.
add_executable(export_tspline_bilinear_shell apps/export_tspline_bilinear_shell.cpp)
target_link_libraries(export_tspline_bilinear_shell PRIVATE mfem_raytracing)

enable_testing()

# Focused data/topology test for the degree-1 T-spline seam workflow.  Kept
# separate from run_tests so the pipe integration fixture remains inspectable.
add_executable(test_tspline_seam_workflow tests/test_tspline_seam_workflow.cpp)
target_link_libraries(test_tspline_seam_workflow PRIVATE mfem_raytracing)
add_test(NAME tspline_seam_workflow COMMAND test_tspline_seam_workflow ${CMAKE_SOURCE_DIR})

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
    tests/test_tspline.cpp
)
target_link_libraries(run_tests PRIVATE mfem_raytracing)

if(MFEM_RAYTRACING_ENABLE_EMBREE)
  target_sources(run_tests PRIVATE
      tests/test_bilinear_intersect.cpp
      tests/test_embree_raytracer.cpp
  )
  target_link_libraries(run_tests PRIVATE mfem_raytracing_embree)

  add_executable(render_leaf_patches apps/render_leaf_patches.cpp)
  target_link_libraries(render_leaf_patches PRIVATE mfem_raytracing_embree mfem_raytracing)

  add_executable(export_scene_json apps/export_scene_json.cpp)
  target_link_libraries(export_scene_json PRIVATE mfem_raytracing_embree mfem_raytracing)

  # Bilinear-seeded vs. naive Newton seeding benchmark (optional local tree).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/intersection_seeding_benchmark/CMakeLists.txt")
    add_subdirectory(intersection_seeding_benchmark)
  endif()

  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(SDL2 QUIET IMPORTED_TARGET sdl2)
  endif()
  if(SDL2_FOUND)
    add_executable(scene_viewer apps/scene_viewer.cpp)
    target_link_libraries(scene_viewer PRIVATE mfem_raytracing_embree mfem_raytracing PkgConfig::SDL2)
  else()
    message(STATUS "SDL2 not found; skipping native scene_viewer")
  endif()
endif()

add_test(NAME unit_tests COMMAND run_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
