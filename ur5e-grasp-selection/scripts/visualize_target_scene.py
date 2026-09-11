#!/usr/bin/env python3
"""
Export Meshcat scenes of a sampled benchmark *target*: the full obstacle scene, the arm at a
goal configuration q_gold, and the target mug welded where that configuration puts it.

This is the first half of a second figure pipeline, parallel to the published one:

    visualize_target_scene.py  ->  out/target_scene*.html   (this file)
    render_target_scene.py     ->  out/target_scene.png     (runs inside Blender)

There is no IK solve here.  `benchmark_boundary_reach.sample_targets` defines a target by
rejection-sampling q_gold ~ U(-pi, pi)^6 until the arm is collision-free in the obstacle
scene and then taking forward kinematics, X_WM = FK_tool0(q_gold) * X_EG.  Every target is
reachable by construction and the arm is clear at it, so posing the figure is just a matter
of drawing one and welding the mug.  The mug hangs in mid-air because the gripper is holding
it; that is the point of the figure.

Two filters are applied to a draw, and both earn their place:

  * The mug must stay below the tops of the shelves, or it floats above the scenery and the
    framing has nothing to sit against.
  * Nothing may come within `--min-clearance` of anything else, the gripper-around-the-mug
    pair excepted.  This closes a real gap rather than being cosmetic: `sample_targets`
    screens the *arm* against a plant that does not contain the target mug at all, so a
    benchmark target is free to put the mug inside a shelf.

Configurations are pinned as files, not as seeds.  The published figure became
unreproducible because only a seed was recorded and branch selection changed underneath it
(see scripts/paper_figure_configuration.json); `--save-configuration` /
`--configuration` are how this pipeline avoids repeating that.

Usage:

    # draw a few candidates, write one HTML + one JSON each, then pick one in a browser
    python scripts/visualize_target_scene.py --no-interactive --seed 42 --num-scenes 4

    # re-pose a pinned one, no sampling at all
    python scripts/visualize_target_scene.py --no-interactive \
        --configuration out/target_scene_02.json --html-out out/target_scene.html
"""
import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from pydrake.all import (  # noqa: E402
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
)
from pydrake.geometry import Meshcat, MeshcatVisualizer  # noqa: E402

from src.ur_experiments import UrIKProblemOldFormulation  # noqa: E402
from scripts.benchmark_boundary_reach import sample_targets  # noqa: E402

SCENE_YAML = os.path.join(REPO_DIR, "models/ur5e_collision_obj.yaml")
ARM_URDF = os.path.join(
    REPO_DIR, "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf")
UR_COLOUR_URDF = os.path.join(
    REPO_DIR, "models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf")
UR_VISUAL_DIR = os.path.join(
    REPO_DIR, "models/universal_robots/ur_description/meshes/ur5e/visual")
MUG_URDF = "package://eaik_ift_experiment/models/mug/mug_simple_red.urdf"

# Half the height of the mug's collision cylinder (length 0.09 in models/mug/mug_simple_red.urdf,
# centred on mug_body_link), so mug_top_z = X_WM.translation()[2] + this.
MUG_HALF_HEIGHT = 0.045

# Top face of a shelf unit.  models/assets/shelves.sdf puts its top board (0.3 x 0.6 x 0.016)
# at local z = +0.3995 and the four shelves* instances are welded at z = 0.4, so the board's
# upper face is at 0.4 + 0.3995 + 0.008.  The side walls (0.783 tall, centred at 0.4) stop
# slightly lower, at 0.7915.
SHELF_TOP_Z = 0.8075

# Pairs allowed to be in contact.  The gripper is supposed to be closed around the mug.
CLEARANCE_EXEMPT_PAIRS = frozenset([frozenset(("wsg", "target_mug"))])


def _fmt(x):
    """Full-precision float for the weld directive and the pinned JSON alike.

    The weld is built by formatting these numbers into a YAML string, so a configuration
    round-trips exactly only if it is saved with the same precision it is welded with.
    """
    return repr(float(x))


def ensure_color_split_artifacts():
    """Generate the per-colour OBJs and colour URDF only if they are missing.

    They are committed to the repository, so this is normally a no-op -- but
    `generate_obj_description_files` re-parses all seven .dae files through trimesh even when
    every output already exists, and trimesh is not a declared dependency of this project.
    Importing it unconditionally would make a clean environment fail for no benefit.
    """
    if os.path.exists(UR_COLOUR_URDF) and glob(os.path.join(UR_VISUAL_DIR, "*.obj")):
        return
    print("[setup] colour-split artifacts missing; regenerating (needs trimesh)")
    from scripts.visualize_grasp_selection import generate_obj_description_files
    generate_obj_description_files()


def build_scene(X_WM=None, meshcat=None):
    """Build the obstacle scene, optionally welding the target mug at X_WM.

    Modelled on benchmark_boundary_reach.build_env, but on models/ur5e_collision_obj.yaml so
    the arm and gripper carry their per-colour visual split through to Blender.  That YAML is
    collision-identical to the benchmark's scene, so the same diagram serves as both the
    collision oracle and the render scene.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(os.path.join(REPO_DIR, "package.xml"))
    parser.package_map().AddPackageXml(
        os.path.join(REPO_DIR, "models/universal_robots/ur_description/package.xml"))
    ProcessModelDirectives(LoadModelDirectives(SCENE_YAML), plant, parser)

    if X_WM is not None:
        xyz = X_WM.translation()
        rpy_deg = np.degrees(X_WM.rotation().ToRollPitchYaw().vector())
        ProcessModelDirectives(LoadModelDirectivesFromString(f"""
directives:
  - add_model:
      name: target_mug
      file: {MUG_URDF}
  - add_weld:
      parent: world
      child: target_mug::mug_body_link
      X_PC:
        translation: [{_fmt(xyz[0])}, {_fmt(xyz[1])}, {_fmt(xyz[2])}]
        rotation: !Rpy {{ deg: [{_fmt(rpy_deg[0])}, {_fmt(rpy_deg[1])}, {_fmt(rpy_deg[2])}] }}
"""), plant, parser)

    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    return builder.Build(), plant, scene_graph


def evaluate_clearance(X_WM, q_gold, max_distance=0.1):
    """Smallest signed distance in the posed scene, and the pair that owns it.

    `max_distance` is a *maximum* -- the query only returns pairs closer than it -- so it has
    to be generous.  Calling this with the acceptance threshold would make the reported
    number bottom out at the threshold and say nothing.

    Pairs internal to one model instance are skipped (the arm's own self-proximity is already
    screened by sample_targets), as is the gripper-versus-mug pair.
    """
    diagram, plant, scene_graph = build_scene(X_WM=X_WM)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("ur5e"), q_gold)

    query = scene_graph.get_query_output_port().Eval(
        scene_graph.GetMyContextFromRoot(context))
    inspector = query.inspector()

    def model_of(geometry_id):
        body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
        return plant.GetModelInstanceName(body.model_instance())

    worst = (np.inf, None)
    for pair in query.ComputeSignedDistancePairwiseClosestPoints(max_distance):
        a, b = model_of(pair.id_A), model_of(pair.id_B)
        if a == b or frozenset((a, b)) in CLEARANCE_EXEMPT_PAIRS:
            continue
        if pair.distance < worst[0]:
            worst = (pair.distance, f"{a} vs {b}")
    return worst


def draw_scenes(args, rng):
    """Rejection-sample accepted targets, reporting the funnel."""
    diagram_ref, _, _ = build_scene()
    old_prob = UrIKProblemOldFormulation(diagram_ref, ARM_URDF)

    accepted, draws, too_high, too_close = [], 0, 0, 0
    t0 = time.time()
    while len(accepted) < args.num_scenes and draws < args.max_draws:
        drawn = sample_targets(old_prob, 1, rng)
        if not drawn:
            break
        _, X_WM, q_gold = drawn[0]
        draws += 1

        mug_top_z = X_WM.translation()[2] + MUG_HALF_HEIGHT
        if mug_top_z > args.max_mug_top_z:
            too_high += 1
            continue

        clearance, worst_pair = evaluate_clearance(X_WM, q_gold)
        if clearance < args.min_clearance:
            too_close += 1
            continue

        accepted.append({
            "q_gold": [float(v) for v in q_gold],
            "X_WM": {
                "translation": [float(v) for v in X_WM.translation()],
                "rpy_deg": [float(v) for v in
                            np.degrees(X_WM.rotation().ToRollPitchYaw().vector())],
            },
            "seed": args.seed,
            "draw_index": draws,
            "min_clearance": float(clearance),
            "closest_pair": worst_pair,
            "mug_top_z": float(mug_top_z),
        })

    print(f"[sample] {draws} draws in {time.time() - t0:.1f}s -> "
          f"{len(accepted)} accepted, {too_high} above z={args.max_mug_top_z} m, "
          f"{too_close} closer than {args.min_clearance} m")
    if len(accepted) < args.num_scenes:
        print(f"[sample] WARNING: wanted {args.num_scenes}, got {len(accepted)}. "
              f"Raise --max-draws, or loosen --min-clearance / --max-mug-top-z.")
    return accepted


def config_to_pose(cfg):
    return RigidTransform(
        RollPitchYaw(np.radians(np.asarray(cfg["X_WM"]["rpy_deg"]))),
        np.asarray(cfg["X_WM"]["translation"]),
    )


def export_scene(meshcat, cfg, html_path, camera_pos, camera_target):
    """Pose one configuration and write a Meshcat static page for it."""
    X_WM = config_to_pose(cfg)
    q_gold = np.asarray(cfg["q_gold"])

    meshcat.Delete()
    diagram, plant, _ = build_scene(X_WM=X_WM, meshcat=meshcat)
    context = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(context),
                       plant.GetModelInstanceByName("ur5e"), q_gold)

    # Browser-side framing only.  The Blender camera is set independently by
    # render_target_scene.py, so this affects the sanity check and nothing else.
    meshcat.SetCameraTarget(camera_target)
    meshcat.SetCameraPose(camera_pos, camera_target)

    diagram.ForcedPublish(context)
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(meshcat.StaticHtml())

    json_path = os.path.splitext(html_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({
            "_comment": [
                "One sampled benchmark target, pinned as a configuration rather than a seed.",
                "Reproduce with:",
                "  python scripts/visualize_target_scene.py --no-interactive \\",
                f"      --configuration {os.path.relpath(json_path, REPO_DIR)}",
                "The weld is built from X_WM.rpy_deg at this precision, so the round trip is",
                "exact.  q_gold is the arm configuration; the mug frame is the grasp frame,",
                "X_WM = FK_tool0(q_gold) * X_EG.",
            ],
            **cfg,
        }, f, indent=2)
    return json_path


def scene_paths(html_out, count):
    """One path when there is one scene, numbered siblings when there are several."""
    if count == 1:
        return [html_out]
    stem, ext = os.path.splitext(html_out)
    return [f"{stem}_{i:02d}{ext}" for i in range(count)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the target draw (default: 42)")
    p.add_argument("--num-scenes", type=int, default=4,
                   help="how many accepted targets to export (default: 4). Each static page "
                        "is tens of MB.")
    p.add_argument("--max-draws", type=int, default=2000,
                   help="give up after this many draws (default: 2000)")
    p.add_argument("--max-mug-top-z", type=float, default=SHELF_TOP_Z,
                   help=f"reject a target whose mug rises above this (default: {SHELF_TOP_Z}, "
                        f"the top face of a shelf unit)")
    p.add_argument("--min-clearance", type=float, default=0.005,
                   help="required margin in metres between every pair of models except "
                        "gripper-vs-target-mug (default: 0.005)")
    p.add_argument("--html-out", default="out/target_scene.html",
                   help="output page; with --num-scenes > 1 this is the stem for "
                        "target_scene_NN.html (default: out/target_scene.html)")
    p.add_argument("--configuration", default=None,
                   help="pose this pinned configuration JSON and skip sampling entirely")
    p.add_argument("--camera-pos", type=float, nargs=3, default=[1.6, -1.3, 1.1],
                   help="Meshcat camera position (browser view only)")
    p.add_argument("--camera-target", type=float, nargs=3, default=[0.25, 0.0, 0.35],
                   help="Meshcat camera target (browser view only)")
    p.add_argument("--no-interactive", action="store_true",
                   help="exit after exporting instead of holding the server open")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_color_split_artifacts()

    rng = np.random.default_rng(args.seed)

    if args.configuration is not None:
        with open(args.configuration) as f:
            configs = [json.load(f)]
        print(f"[config] posing pinned {args.configuration}, no sampling")
    else:
        configs = draw_scenes(args, rng)
        if not configs:
            raise SystemExit("No target survived the filters; nothing to export.")

    meshcat = Meshcat()
    print(f"[meshcat] {meshcat.web_url()}")

    paths = scene_paths(args.html_out, len(configs))
    rows = []
    for cfg, html_path in zip(configs, paths):
        json_path = export_scene(meshcat, cfg, html_path,
                                 args.camera_pos, args.camera_target)
        rows.append((html_path, json_path, cfg))

    print()
    print(f"{'scene':<28} {'mug xyz':<26} {'clear':>8}  closest pair")
    for html_path, _, cfg in rows:
        xyz = cfg["X_WM"]["translation"]
        print(f"{os.path.basename(html_path):<28} "
              f"[{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}]      "
              f"{cfg['min_clearance']:8.4f}  {cfg['closest_pair']}")
    print()
    for _, _, cfg in rows:
        print("  q_gold = [" + ", ".join(f"{v:+.6f}" for v in cfg["q_gold"]) + "]")
    print()
    print("Open the pages in a browser, pick one, then render it:")
    print(f"  python scripts/visualize_target_scene.py --no-interactive \\")
    print(f"      --configuration {rows[0][1]} --html-out {args.html_out}")
    print(f"  ./scripts/render_target_scene.sh -- --camera-azimuth 95 --camera-elevation 25")

    if args.no_interactive:
        return

    meshcat.AddButton("Sample Another Target")
    clicks = meshcat.GetButtonClicks("Sample Another Target")
    print("\nPress 'Sample Another Target' in the Meshcat panel; Ctrl-C to stop.")
    try:
        while True:
            new_clicks = meshcat.GetButtonClicks("Sample Another Target")
            if new_clicks > clicks:
                clicks = new_clicks
                fresh = draw_scenes(argparse.Namespace(**{**vars(args), "num_scenes": 1}), rng)
                if fresh:
                    export_scene(meshcat, fresh[0], args.html_out,
                                 args.camera_pos, args.camera_target)
                    xyz = fresh[0]["X_WM"]["translation"]
                    print(f"  mug at [{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}], "
                          f"clearance {fresh[0]['min_clearance']:.4f} m -> {args.html_out}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
