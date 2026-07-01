cmake_minimum_required(VERSION 3.16)
project(mfem_raytracing CXX)
set(CMAKE_CXX_STANDARD 17)

set(MFEM_DIR "$ENV{HOME}/mesh_software/mfem-4.9/build" CACHE PATH "MFEM build directory")
list(APPEND CMAKE_PREFIX_PATH "${MFEM_DIR}")
find_package(mfem REQUIRED)

add_library(mfem_raytracing
    src/intersect_AABB.cpp
    src/cartesian_mesh_builder.cpp
    src/cartesian_mesh_naming.cpp
    src/ray_tracing.cpp
    src/nurbs_mesh_operations.cpp
    src/bilinear_intersection.cpp
    src/element_extractor.cpp
    src/kronecker_product.cpp
    src/ien_generator.cpp
    src/id_generator.cpp
    src/lm_generator.cpp
    src/bdr_patch_extractor.cpp
    src/bezier_degree_reduction.cpp
    src/b_spline_curve_reduction.cpp
    src/nurbs_degree_reduction.cpp
)
target_include_directories(mfem_raytracing PUBLIC include)
target_link_libraries(mfem_raytracing PUBLIC mfem)

add_executable(make_cart_mesh src/make_cart_mesh.cpp)
target_link_libraries(make_cart_mesh PRIVATE mfem_raytracing)

add_executable(print_bdr_elements tests/print_bdr_elements.cpp)
target_link_libraries(print_bdr_elements PRIVATE mfem)

add_executable(print_bdr_patch_blocks tests/print_bdr_patch_blocks.cpp)
target_link_libraries(print_bdr_patch_blocks PRIVATE mfem_raytracing)

enable_testing()

add_executable(run_tests
    tests/main.cpp
    tests/test_ray.cpp
    tests/test_intersect_aabb.cpp
    tests/test_cartesian_mesh.cpp
    tests/test_ray_trace.cpp
    tests/test_nurbs_mesh.cpp
    tests/test_bilinear_intersection.cpp
    tests/test_element_extractor.cpp
    tests/test_kronecker_product.cpp
    tests/test_ien_generator.cpp
    tests/test_id_generator.cpp
    tests/test_lm_generator.cpp
    tests/test_iga_reference_connectivity.cpp
    tests/test_bdr_patch_extractor.cpp
    tests/test_b_spline_curve_reduction.cpp
    tests/test_bezier_degree_reduction.cpp
    tests/test_nurbs_degree_reduction.cpp
)
target_link_libraries(run_tests PRIVATE mfem_raytracing)
add_test(NAME unit_tests COMMAND run_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
