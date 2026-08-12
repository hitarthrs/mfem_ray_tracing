# Test JSON fixtures

Canonical leaf/catalog inputs for C++ unit and integration tests.
Paths in tests are relative to the repo root (`CMAKE_SOURCE_DIR`).

| File | Tests |
| --- | --- |
| `d4_leaf_bboxes.json` | `test_surface_multistep_reduction`, `test_embree_raytracer` |
| `pipe_nurbs_border_patches.json` | `test_tspline_seam_workflow` (catalog) |
| `pipe_patch_all_e_0_05_hard_seams.json` | `test_tspline_seam_workflow` (hard-seam leaves) |
| `all_patches_0_05.json` | `test_tspline_seam_workflow` (collar / direct path) |

Regenerate from `python_experiments/multiple_step_degree_reduction_surfaces/` demos if goldens change.
