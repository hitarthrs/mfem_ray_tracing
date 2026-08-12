# Bilinear-seeded vs. naive Newton-Raphson seeding benchmark.
#
# Added from the top-level CMakeLists only when MFEM_RAYTRACING_ENABLE_EMBREE is
# ON (the bilinear seed uses the Embree BVH). Links the core reduction library
# (leaf reduction + LoadSurfaceDataJson) and the Embree ray tracer.

add_library(seeding_benchmark_lib
    src/nurbs_surface.cpp
    src/newton_intersect.cpp
    src/bilinear_seeder.cpp
)
target_include_directories(seeding_benchmark_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(seeding_benchmark_lib
    PUBLIC mfem_raytracing mfem_raytracing_embree)

add_executable(seeding_benchmark src/benchmark_main.cpp)
target_link_libraries(seeding_benchmark PRIVATE seeding_benchmark_lib)
