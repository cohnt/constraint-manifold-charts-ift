"""Generate meshcat HTML for the constrained-IRIS segment of the overview video.

Bimanual IIWA holding a fixed gripper-to-gripper transform, with the table but
no shelves, and a **constrained** IRIS-NP2 region: IRIS runs in the 8-D
parameterized space (7 left-arm joints plus the redundancy angle psi), with the
analytic-IK parameterization supplied to `IrisNp2Options.parameterization`, and
with the reachability and joint-limit constraints attached to its
`prog_with_additional_constraints`. The segment then random-walks inside that
region, lifting each sample through the same parameterization to pose the robot.

That is the whole point of the shot: every configuration shown satisfies the
bimanual constraint *by construction*, because the region lives in the
parameterized space rather than in the 14-D configuration space.

## What this used to do, and why it was wrong

The previous version defined a `parameterization` function and then never
passed it to IRIS. It called `IrisNp2` with a 14-D box domain and a full-space
seed, so it grew an ordinary configuration-space region while the docstring and
the video captioned it as a region "in the parameterized space". It also fell
back to a hand-made `HPolyhedron.MakeBox` around the seed if IRIS raised, and
carried on presenting that box as an IRIS region. Both are the failure this
repository has hit before: a visualization that does not compute what its
caption claims. There is no silent fallback here now -- if IRIS fails, so does
this script.

The setup mirrors `run_iris_and_gcs` in the IIWA experiment's
`scripts/experiments/run_full_comparison.py`, which is the authoritative
implementation; the constraint objects come from that repo's C++ extension, so
it must be built (see ../docs/INSTALL.md).

Usage:
    .venv/bin/python scripts/video/generate_iiwa_iris_meshcat.py
"""

import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_REPO = os.path.join(REPO, os.pardir, "iiwa-bimanual")
# The IIWA experiment folder (its own modules import as `src.foo`), its src/ (ours
# import `iiwa_analytic_ik` flat) and the built C++ extension.
sys.path.insert(0, IIWA_REPO)
sys.path.insert(0, os.path.join(IIWA_REPO, "src"))
sys.path.insert(0, os.path.join(IIWA_REPO, "cpp_parameterization", "python"))

from iiwa_ik import (  # noqa: E402
    AutoDiffConfig, BimanualConfig, IftSingularityHandling,
    IiwaBimanualJointLimitConstraint, MakeParameterization, ReachabilityType,
)
from src.reach_constraints import make_reach_constraint  # noqa: E402
from iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper  # noqa: E402

from pydrake.all import (  # noqa: E402
    CollisionCheckerParams, Hyperellipsoid, HPolyhedron, IrisNp2,
    IrisNp2Options, LoadModelDirectives, MathematicalProgram, MeshcatVisualizer,
    MeshcatVisualizerParams, Parser, ProcessModelDirectives, RandomGenerator,
    RobotDiagramBuilder, Role, SceneGraphCollisionChecker, SnoptSolver,
    StartMeshcat,
)

DIRECTIVES = os.path.join(REPO, "models", "iiwa_bimanual_table_only.dmd.yaml")
OUT_HTML = os.path.join(REPO, "video", "v2_iiwa_iris.html")

FPS = 30
# ~10 s, which is what the segment is cut to. 10 x 30 = 300 frames.
N_WALK_STEPS = 10
INTERP_STEPS = 30

GRASP_DISTANCE = 0.6
BOUNDARY_THRESHOLD = 2.46
BOUNDARY_EPSILON = 1e-6

# A seed in the 8-D parameterized space, from the IIWA experiment's SEEDS list
# (Q_TILDE_MIDDLE). Valid here too: this scene is the shelves scene minus an
# obstacle, so anything collision-free there is collision-free here.
Q_TILDE_SEED = np.array([
    -0.5997312520566763, 1.489780849654964, -1.4739679827359913,
    1.2905366081785483, -0.04421061906813227, -0.8793712572715165,
    -1.1603461715511334, 1.45,
])

DOMAIN_LOWER = np.hstack((iiwa_limits_lower, [0.0]))
DOMAIN_UPPER = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))


def build_checker(meshcat):
    """Collision checker + diagram for the table-only bimanual scene."""
    params = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.0)

    # Illustration role only; the proximity tree must never reach meshcat or it
    # ends up in the StaticHtml export and then in the Blender render.
    vis_params = MeshcatVisualizerParams()
    vis_params.delete_on_initialization_event = False
    vis_params.role = Role.kIllustration
    vis_params.prefix = "visual"
    MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, vis_params)

    plant = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(IIWA_REPO, "package.xml"))
    ProcessModelDirectives(LoadModelDirectives(DIRECTIVES), parser)

    # Both arms AND both grippers are robot instances. With the grippers left
    # out they count as environment, and a CollisionChecker never checks
    # environment-against-environment pairs -- so gripper-vs-table collisions
    # would be invisible and the region would be certified through the table.
    params.robot_model_instances = [
        plant.GetModelInstanceByName(n)
        for n in ("iiwa_left", "iiwa_right", "wsg_left", "wsg_right")
    ]
    plant.Finalize()

    builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
    builder.builder().ExportOutput(plant.get_state_output_port(), "state")

    diagram = builder.Build()
    params.model = diagram
    params.edge_step_size = 0.01
    return SceneGraphCollisionChecker(params), diagram, plant


def make_iris_options():
    """IrisNp2Options carrying the parameterization and the extra constraints.

    Same configuration as run_iris_and_gcs in the IIWA experiment: the IFT is used
    for the parameterization's derivatives, reachability is the probing
    formulation, and the joint-limit constraint is enforced on the *lifted*
    configuration rather than on the parameters.
    """
    config = BimanualConfig(
        shoulder_up=True, elbow_up=True, wrist_up=False,
        grasp_distance=GRASP_DISTANCE,
        clipping_margin=1e-4, clipping_margin_psi=1e-4,
        boundary_threshold=BOUNDARY_THRESHOLD, boundary_epsilon=BOUNDARY_EPSILON,
    )
    ad_config = AutoDiffConfig(
        use_ift=True, ift_handling=IftSingularityHandling.kPseudoinverse,
        lambda_=0.0,
    )

    options = IrisNp2Options()
    options.parameterization = MakeParameterization(config, ad_config)
    options.sampled_iris_options.random_seed = 2
    options.sampled_iris_options.max_iterations = 1
    options.sampled_iris_options.relax_margin = True
    options.sampled_iris_options.epsilon = 0.01
    options.sampled_iris_options.delta = 0.01
    options.sampled_iris_options.sample_particles_in_parallel = False
    options.add_hyperplane_if_solve_fails = True
    options.solver_options.SetOption(
        SnoptSolver().solver_id(), "Major iterations limit", 200)

    prog = MathematicalProgram()
    q_tilde = prog.NewContinuousVariables(8, "q_tilde")
    options.sampled_iris_options.prog_with_additional_constraints = prog
    reach_con = make_reach_constraint(ReachabilityType.kProbing, config,
                                      ad_config)
    prog.AddConstraint(reach_con, q_tilde)
    jl_con = IiwaBimanualJointLimitConstraint(
        iiwa_limits_lower, iiwa_limits_upper, config, ad_config)
    prog.AddConstraint(jl_con, q_tilde)

    # `prog` (and the constraints bound into it) are handed to IRIS as a raw
    # pointer via prog_with_additional_constraints, which does NOT keep them
    # alive. Letting them fall out of scope here segfaults inside IrisNp2, so
    # they are returned and held by the caller for the duration of the call.
    keepalive = (prog, reach_con, jl_con, config, ad_config)
    return options, config, ad_config, keepalive


def main():
    print("Starting meshcat...")
    meshcat = StartMeshcat()

    print("Building the bimanual scene (table, no shelves)...")
    checker, diagram, plant = build_checker(meshcat)

    options, config, ad_config, _iris_keepalive = make_iris_options()
    lift = MakeParameterization(config, ad_config).get_parameterization_double()

    q_full_seed = np.asarray(lift(Q_TILDE_SEED)).flatten()
    if not checker.CheckConfigCollisionFree(q_full_seed):
        raise RuntimeError(
            "the IRIS seed lifts to a configuration that is in collision; "
            "IRIS cannot grow a region from it")

    print("Growing a constrained IRIS-NP2 region in the 8-D parameterized "
          "space...")
    t0 = time.time()
    domain = HPolyhedron.MakeBox(DOMAIN_LOWER, DOMAIN_UPPER)
    # No try/except. A region that failed to grow and was quietly replaced by a
    # box is indistinguishable on screen from one that succeeded.
    region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(1e-2, Q_TILDE_SEED),
                     domain, options).ReduceInequalities()
    print(f"  {region.b().shape[0]} halfplanes in {region.ambient_dimension()}-D "
          f"({time.time() - t0:.1f}s)")
    if region.ambient_dimension() != 8:
        raise RuntimeError(
            f"IRIS returned a {region.ambient_dimension()}-D region; the "
            "parameterization was not applied, so this is a configuration-space "
            "region and must not be shown as a constrained one")

    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
    recorder = diagram.GetSubsystemByName("meshcat_visualizer(visual)")

    meshcat.SetCameraPose(camera_in_world=[1.8, -0.8, 1.4],
                          target_in_world=[0.0, 0.38, 0.5])

    print(f"Random walk inside the region: {N_WALK_STEPS} steps x "
          f"{INTERP_STEPS} frames...")
    recorder.StartRecording(set_transforms_while_recording=False)

    rng = RandomGenerator(42)
    q_tilde_cur = Q_TILDE_SEED.copy()
    frame_idx = 0
    n_checked = 0

    for step in range(N_WALK_STEPS):
        q_tilde_next = region.UniformSample(rng, q_tilde_cur)

        for j in range(INTERP_STEPS):
            alpha = (j + 1) / INTERP_STEPS
            q_tilde = (1 - alpha) * q_tilde_cur + alpha * q_tilde_next
            # The region is convex, so the interpolant is inside it too; the
            # lift is what turns a parameter into a full configuration.
            q_full = np.asarray(lift(q_tilde)).flatten()
            if checker.CheckConfigCollisionFree(q_full):
                n_checked += 1

            plant.SetPositions(plant_ctx, q_full)
            ctx.SetTime(frame_idx / FPS)
            diagram.ForcedPublish(ctx)
            frame_idx += 1

        q_tilde_cur = q_tilde_next

    recorder.StopRecording()
    recorder.PublishRecording()

    print(f"  {frame_idx} frames, {frame_idx / FPS:.1f}s; "
          f"{n_checked}/{frame_idx} collision-free")
    if n_checked < frame_idx:
        print("  WARNING: some shown configurations are in collision. The "
              "region is only probabilistically certified, but a large count "
              "here means the parameterization and the checker disagree.")

    print("Exporting StaticHtml...")
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(meshcat.StaticHtml())
    print(f"Wrote {OUT_HTML} ({os.path.getsize(OUT_HTML) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
