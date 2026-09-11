"""Export per-visual-geometry world transforms for each frame.

Computes X_WG = X_WB @ X_BG for every visual geometry at every frame.
Also records which mesh file each geometry uses.

Usage:
    .venv/bin/python scripts/video/export_body_transforms.py --point 0
"""

import argparse
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from pydrake.multibody.tree import BodyIndex
from pydrake.math import RigidTransform
from rby1_planning import Rby1ActiveJointLayout, make_default_rby1_infrastructure
import plan_grid as E
from plan_format.plan_io import load_plan


def extract_visual_geometries(plant, scene_graph, context):
    """Get visual geometry info: body, mesh source, and X_BG offset."""
    inspector = scene_graph.model_inspector()
    vis_geos = []

    for geo_id in inspector.GetAllGeometryIds():
        props = inspector.GetIllustrationProperties(geo_id)
        if not props:
            continue
        frame_id = inspector.GetFrameId(geo_id)
        body = plant.GetBodyFromFrameId(frame_id)
        X_BG = inspector.GetPoseInFrame(geo_id)
        shape = inspector.GetShape(geo_id)
        shape_name = type(shape).__name__

        mesh_path = None
        try:
            src = shape.source()
            mesh_path = str(src)
            if "path='" in mesh_path:
                mesh_path = mesh_path.split("path='")[1].rstrip("')")
        except AttributeError:
            pass

        vis_geos.append({
            "body_name": body.name(),
            "body_index": int(body.index()),
            "shape_type": shape_name,
            "mesh_path": str(mesh_path) if mesh_path else None,
            "mesh_basename": os.path.basename(str(mesh_path)) if mesh_path else None,
            "X_BG": X_BG.GetAsMatrix4().tolist(),
        })

    return vis_geos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(REPO, "video", f"transforms_point{args.point:02d}.pkl")

    print(f"Loading plan for point {args.point}...")
    plan_path = os.path.join(REPO, "plans", "grid_cache", f"point_{args.point:02d}.pkl")
    _, meta = load_plan(plan_path)
    legs = meta["legs"]

    print("Building Drake infrastructure...")
    bx, by = meta["bx"], meta["by"]
    obstacles = [E.TABLE] + E.open_box_walls(E.box_pose_for(bx, by))
    plant, checker, diagram = make_default_rby1_infrastructure(
        obstacles=obstacles, open_grippers=True)
    layout = Rby1ActiveJointLayout(plant)

    sg = diagram.GetSubsystemByName("scene_graph")
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    sg_ctx = sg.GetMyContextFromRoot(ctx)

    vis_geos = extract_visual_geometries(plant, sg, ctx)
    print(f"  {len(vis_geos)} visual geometries")

    all_frames = []
    for leg in legs:
        qs = leg["q"]
        for q23 in qs:
            q_full = plant.GetPositions(pctx)
            q_full[layout.plant_idxs] = q23
            plant.SetPositions(pctx, q_full)

            frame_data = []
            for vg in vis_geos:
                body = plant.get_body(BodyIndex(vg["body_index"]))
                X_WB = plant.EvalBodyPoseInWorld(pctx, body)
                X_BG = RigidTransform(np.array(vg["X_BG"]))
                X_WG = X_WB @ X_BG
                frame_data.append({
                    "X_WB": X_WB.GetAsMatrix4().tolist(),
                    "X_WG": X_WG.GetAsMatrix4().tolist(),
                })

            all_frames.append(frame_data)

    total_dur = sum(l["duration"] for l in legs)
    print(f"  {len(all_frames)} frames, {total_dur:.1f}s total")

    result = {
        "point": args.point,
        "visual_geometries": vis_geos,
        "frames": all_frames,
        "fps": 30,
        "legs": [{"name": l["name"], "n_frames": len(l["q"]), "duration": l["duration"]} for l in legs],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
