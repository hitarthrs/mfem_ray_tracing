#!/usr/bin/env python3
"""Split the pipe elbow boundary patches into per-patch SurfaceData JSON files.

Reads python_experiments/multiple_step_degree_reduction_surfaces/
pipe_nurbs_border_patches.json (16 MFEM NURBS boundary patches of the thick-walled
90 deg elbow) and writes, into ./surfaces/:

  * patch_<id>_<role>.json  -- one file per patch, in the schema consumed by the
    C++ LoadSurfaceDataJson() (name, degree_u/v, control_points[nu][nv][3],
    weights (null|[nu][nv]), knotvector_u, knotvector_v).
  * manifest.csv            -- patch_id, role, group, name, json path.

The stratification group is derived from the role:
  * inlet_endcap / outlet_endcap -> "flat"  (planar annular quarter-caps)
  * inner_wall   / outer_wall    -> "bend"  (curved elbow walls, high curvature)

Run from the repository root:
    python3 intersection_seeding_benchmark/prepare_pipe_surfaces.py
"""

import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BORDER_JSON = os.path.join(
    REPO_ROOT,
    "python_experiments",
    "multiple_step_degree_reduction_surfaces",
    "pipe_nurbs_border_patches.json",
)
OUT_DIR = os.path.join(REPO_ROOT, "intersection_seeding_benchmark", "surfaces")

FLAT_ROLES = {"inlet_endcap", "outlet_endcap"}
BEND_ROLES = {"inner_wall", "outer_wall"}


def group_for_role(role: str) -> str:
    if role in FLAT_ROLES:
        return "flat"
    if role in BEND_ROLES:
        return "bend"
    raise ValueError(f"unknown role {role!r}")


def main() -> None:
    with open(BORDER_JSON) as fh:
        doc = json.load(fh)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_rows = []

    for patch in doc["patches"]:
        pid = patch["id"]
        role = patch["role"]
        group = group_for_role(role)
        name = patch["name"]

        surface = {
            "name": name,
            "degree_u": patch["degree_u"],
            "degree_v": patch["degree_v"],
            "control_points": patch["control_points"],
            "weights": patch.get("weights"),
            "knotvector_u": patch["knotvector_u"],
            "knotvector_v": patch["knotvector_v"],
        }

        fname = f"patch_{pid:02d}_{role}.json"
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "w") as fh:
            json.dump(surface, fh, indent=2)

        manifest_rows.append(
            {
                "patch_id": pid,
                "role": role,
                "group": group,
                "name": name,
                # path relative to repo root, for a stable CWD-independent handle
                "json": os.path.relpath(fpath, REPO_ROOT),
            }
        )

    manifest_rows.sort(key=lambda r: r["patch_id"])
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["patch_id", "role", "group", "name", "json"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {len(manifest_rows)} patch surfaces to {OUT_DIR}")
    print(f"wrote manifest {manifest_path}")
    n_flat = sum(1 for r in manifest_rows if r["group"] == "flat")
    n_bend = sum(1 for r in manifest_rows if r["group"] == "bend")
    print(f"  flat group: {n_flat} patches   bend group: {n_bend} patches")


if __name__ == "__main__":
    main()
