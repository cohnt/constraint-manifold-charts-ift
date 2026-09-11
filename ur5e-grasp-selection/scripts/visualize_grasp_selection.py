#!/usr/bin/env python3
import os
import sys
import time
import argparse
import copy
import json

# Put the folder root on sys.path so `import src.*` resolves however this script
# is invoked. Its siblings already do this; without it, running the documented
# `./scripts/reproduce_grasp_figure.sh` fails with ModuleNotFoundError: 'src'.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

import numpy as np


from pydrake.geometry import MeshcatVisualizer, Meshcat
import time

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    RgbdSensor,
    RenderCameraCore,
    ColorRenderCamera,
    DepthRenderCamera,
    CameraInfo,
    ClippingRange,
    DepthRange,
    Role,
    RoleAssign,
    IllustrationProperties,
    PerceptionProperties,
    MakeRenderEngineVtk,
    RenderEngineVtkParams
)

from src.util import RepoDir
from src.eaik_ik import EaikIK
from src.ur_experiments import UrProblemOptions, UrIKProblemNewFormulation


def convert_mesh_if_needed(src_path, dst_path, force_transform=None):
    if os.path.exists(dst_path):
        return
    print(f"Converting mesh: {src_path} -> {dst_path}")
    import trimesh
    scene = trimesh.load(src_path)
    if isinstance(scene, trimesh.Scene):
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            geom = scene.geometry[geometry_name].copy()
            geom.apply_transform(transform)
            if force_transform is not None:
                geom.apply_transform(force_transform)
            meshes.append(geom)
        if len(meshes) > 0:
            mesh = trimesh.util.concatenate(meshes)
            _ = mesh.vertex_normals
        else:
            print(f"WARNING: No geometry found in {src_path}")
            return
    else:
        mesh = scene
        if force_transform is not None:
            mesh.apply_transform(force_transform)
        _ = mesh.vertex_normals
    mesh.export(dst_path)
    
    # Strip material references so Drake VTK uses the dynamically assigned perception properties
    with open(dst_path, "r") as f:
        lines = f.readlines()
    with open(dst_path, "w") as f:
        for line in lines:
            if not line.startswith("mtllib") and not line.startswith("usemtl"):
                f.write(line)


def split_dae_by_color(src_path, out_dir, base_name):
    """
    Convert one UR5e visual `.dae` into one OBJ per distinct material colour.

    The `.dae` meshes carry the real UR look as four colours bound per sub-geometry
    (light-grey shells, dark-grey joint bands, near-black trim, UR blue accents).  Two
    hard constraints force this split:

      * Drake cannot read `.dae` at all, which is why the OBJ conversion exists.  Blender
        5.0.1 cannot read it either (`bpy.ops.wm.collada_import` fails to poll -- the
        OpenCollada importer was removed), so recovering the colours downstream is not an
        option; they have to survive the Drake conversion.
      * Meshcat sends exactly one material per geometry.  A single OBJ per link therefore
        cannot carry four colours, no matter how the OBJ is written.

    So sub-meshes are grouped by colour (forearm's 7 sub-meshes collapse to 4 groups) and
    each group is exported as its own OBJ.  The colour itself rides in the generated URDF
    `<material>` rather than in the OBJ, which keeps Drake off its VTK material path;
    hence the `mtllib`/`usemtl` strip below stays.

    Returns [(obj_filename, (r, g, b)), ...] ordered deterministically by colour.
    """
    import trimesh
    scene = trimesh.load(src_path)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    groups = {}  # rgb hex -> list of transformed meshes
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geom = scene.geometry[geometry_name].copy()
        geom.apply_transform(transform)

        color = (204, 204, 204, 255)
        material = getattr(geom.visual, "material", None)
        for attribute in ("main_color", "baseColorFactor", "diffuse"):
            value = getattr(material, attribute, None) if material is not None else None
            if value is not None:
                value = np.asarray(value).ravel()
                if value.size >= 3:
                    if value.dtype.kind == "f":
                        value = np.clip(value, 0.0, 1.0) * 255.0
                    color = tuple(int(round(c)) for c in value[:3]) + (255,)
                    break
        key = "%02x%02x%02x" % color[:3]
        groups.setdefault(key, []).append(geom)

    outputs = []
    for key in sorted(groups):
        meshes = groups[key]
        dst_name = f"{base_name}_{key}.obj"
        dst_path = os.path.join(out_dir, dst_name)
        rgb = tuple(int(key[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        outputs.append((dst_name, rgb))
        if os.path.exists(dst_path):
            continue
        print(f"Converting mesh: {src_path} [#{key}] -> {dst_path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        _ = mesh.vertex_normals
        mesh.export(dst_path)
        with open(dst_path, "r") as f:
            lines = f.readlines()
        with open(dst_path, "w") as f:
            for line in lines:
                if not line.startswith("mtllib") and not line.startswith("usemtl"):
                    f.write(line)
        # trimesh drops a companion material.mtl next to the OBJ; with the references
        # stripped it is dead weight, and it would otherwise land in git.
        stray_mtl = os.path.join(out_dir, "material.mtl")
        if os.path.exists(stray_mtl):
            os.remove(stray_mtl)
    return outputs


def generate_obj_description_files():
    repo_dir = RepoDir()
    ur_meshes_dir = os.path.join(repo_dir, "models/universal_robots/ur_description/meshes/ur5e/visual")
    ur_mesh_names = ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]
    mesh_groups = {}
    for name in ur_mesh_names:
        src = os.path.join(ur_meshes_dir, f"{name}.dae")
        mesh_groups[name] = split_dae_by_color(src, ur_meshes_dir, name)

    src_urdf = os.path.join(repo_dir, "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf")
    dst_urdf = os.path.join(repo_dir, "models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf")
    if not os.path.exists(dst_urdf):
        print(f"Generating URDF copy: {dst_urdf}")
        with open(src_urdf, "r") as f:
            content = f.read()
        content = expand_visuals_by_color(content, mesh_groups)
        with open(dst_urdf, "w") as f:
            f.write(content)


def expand_visuals_by_color(urdf_text, mesh_groups):
    """
    Rewrite each single-mesh `<visual>` into one `<visual>` per colour group.

    Only `<visual>` blocks are touched -- collision geometry, inertia and joint origins
    are left byte-identical, which the kinematic tests depend on.  The blanket
    `<material name="LightGrey">` is dropped: it was overriding the mesh colours with a
    flat grey, which is half of why the arms rendered white.

    URDF materials are global by name, so each colour gets one name derived from its RGB
    hex.  Reusing a name across links is then safe by construction: identical name implies
    identical colour.
    """
    import re

    def replace(match):
        block = match.group(0)
        indent = match.group("indent")
        stem = match.group("stem")
        groups = mesh_groups.get(stem)
        if not groups:
            return block
        origin = re.search(r"<origin[^>]*/>", block)
        origin_text = origin.group(0) if origin else '<origin rpy="0 0 0" xyz="0 0 0"/>'
        visuals = []
        for obj_name, rgb in groups:
            visuals.append(
                f"{indent}<visual>\n"
                f"{indent}  {origin_text}\n"
                f"{indent}  <geometry><mesh filename=\"package://ur_description/meshes/ur5e/visual/{obj_name}\"/></geometry>\n"
                f"{indent}  <material name=\"URColor_{obj_name.rsplit('_', 1)[1][:-4]}\">"
                f"<color rgba=\"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f} 1.0\"/></material>\n"
                f"{indent}</visual>"
            )
        return "\n".join(visuals)

    pattern = re.compile(
        r"(?P<indent>[ \t]*)<visual>.*?visual/(?P<stem>\w+)\.dae.*?</visual>",
        re.DOTALL,
    )
    new_text, count = pattern.subn(replace, urdf_text)
    if count != len(mesh_groups):
        raise RuntimeError(
            f"Expected to rewrite {len(mesh_groups)} <visual> blocks, rewrote {count}."
        )
    return new_text

def get_initial_guess_for_angle(plant, plant_context, scene_graph, diagram_context, theta_base, seed):
    np.random.seed(seed)
    repo_dir = RepoDir()
    urdf_path = os.path.join(repo_dir, "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf")
    ik_solver = EaikIK(urdf_path)
    
    arm_instance = plant.GetModelInstanceByName("ur5e")
    limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
    limits_upper = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])
    
    for attempt in range(5000):
        z_val = np.random.uniform(0.15, 0.22)
        d_grasp = np.random.uniform(0.15, 0.22)
        theta = theta_base + np.random.uniform(-0.15, 0.15)
        
        v_x = np.array([np.sin(theta), -np.cos(theta), 0.0])
        v_y = np.array([0.0, 0.0, 1.0])
        v_z = np.array([-np.cos(theta), -np.sin(theta), 0.0])
        
        R_WE = RotationMatrix(np.column_stack((v_x, -v_z, v_y)))
        p_WE = np.array([0.38 + d_grasp * np.cos(theta), 0.0 + d_grasp * np.sin(theta), z_val])
        X_WE = RigidTransform(R_WE, p_WE)
        
        sols = ik_solver.solve(X_WE.GetAsMatrix4())
        if not sols: continue
            
        valid_sols = []
        for q in sols:
            q = q.flatten()
            if np.all(q >= limits_lower) and np.all(q <= limits_upper):
                valid_sols.append(q)
                
        if not valid_sols: continue
            
        for q in valid_sols:
            plant.SetPositions(plant_context, arm_instance, q)
            query_object = scene_graph.get_query_output_port().Eval(
                scene_graph.GetMyContextFromRoot(diagram_context)
            )
            dist = query_object.ComputeSignedDistancePairwiseClosestPoints(0.001)
            if min([d.distance for d in dist] + [1.0]) > 0.001:
                return q, theta, d_grasp, z_val, X_WE
                
    return np.zeros(6), theta_base, 0.20, 0.10, RigidTransform()

def set_transparency_of_models(plant, model_names, alpha, scene_graph, arm_alpha=None):
    """
    Set per-model opacity.  The mug stays opaque -- it is the subject.  The arms and
    grippers are drawn semi-transparent so three configurations converging on the same
    small object all remain readable; with opaque arms the mug is buried from every
    camera angle.  Pass arm_alpha=1.0 to keep the arms solid.

    This writes the **alpha channel only**.  The UR links now carry their real per-colour
    materials from the generated URDF (see split_dae_by_color), and flattening the RGB
    here would throw them straight back away.  The mug's forced red is the one deliberate
    RGB override.
    """
    from pydrake.all import Rgba
    if arm_alpha is None:
        arm_alpha = alpha
    inspector = scene_graph.model_inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        frame_id = inspector.GetFrameId(geometry_id)
        frame_name = inspector.GetName(frame_id)

        match = False
        for name in model_names:
            if frame_name == name or frame_name.startswith(name + "::"):
                match = True
                break

        if match:
            if "target_mug" in frame_name:
                target_alpha = 1.0
            elif "ur5e" in frame_name:
                target_alpha = arm_alpha
            else:
                target_alpha = alpha
            
            # Illustration Properties
            properties = inspector.GetIllustrationProperties(geometry_id)
            if properties is None:
                continue # Skip collision geometries
                
            properties_copy = IllustrationProperties(properties)
                
            if properties_copy.HasProperty("phong", "diffuse"):
                phong = properties_copy.GetProperty("phong", "diffuse")
                if "target_mug" in frame_name:
                    phong.set(1.0, 0.0, 0.0, target_alpha)
                else:
                    phong.set(phong.r(), phong.g(), phong.b(), target_alpha)
                properties_copy.UpdateProperty("phong", "diffuse", phong)
            else:
                if "target_mug" in frame_name:
                    phong = Rgba(1.0, 0.0, 0.0, target_alpha)
                elif "ur5e" in frame_name:
                    # Should not happen: every UR visual carries a <material> from the
                    # generated URDF.  If it does, the colour plumbing broke.
                    print(f"WARNING: no diffuse colour on {frame_name}; falling back to grey")
                    phong = Rgba(0.5, 0.5, 0.5, target_alpha)
                else:
                    phong = Rgba(1.0, 1.0, 1.0, target_alpha)
                properties_copy.AddProperty("phong", "diffuse", phong)
            scene_graph.AssignRole(plant.get_source_id(), geometry_id, properties_copy, RoleAssign.kReplace)
            
            # Perception Properties
            perc_properties = inspector.GetPerceptionProperties(geometry_id)
            if perc_properties is None:
                continue
                
            perc_properties_copy = PerceptionProperties(perc_properties)
                
            if perc_properties_copy.HasProperty("phong", "diffuse"):
                phong = perc_properties_copy.GetProperty("phong", "diffuse")
                if "target_mug" in frame_name:
                    phong.set(1.0, 0.0, 0.0, target_alpha)
                else:
                    phong.set(phong.r(), phong.g(), phong.b(), target_alpha)
                perc_properties_copy.UpdateProperty("phong", "diffuse", phong)
            else:
                if "target_mug" in frame_name:
                    phong = Rgba(1.0, 0.0, 0.0, target_alpha)
                elif "ur5e" in frame_name:
                    phong = Rgba(0.5, 0.5, 0.5, target_alpha)
                else:
                    phong = Rgba(1.0, 1.0, 1.0, target_alpha)
                perc_properties_copy.AddProperty("phong", "diffuse", phong)
                
            scene_graph.RemoveRole(plant.get_source_id(), geometry_id, Role.kPerception)
            scene_graph.AssignRole(plant.get_source_id(), geometry_id, perc_properties_copy)

def setup_camera_look_at(camera_pos, target_pos):
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis /= np.linalg.norm(z_axis)
    world_up = np.array([0, 0, 1])
    x_axis = np.cross(z_axis, world_up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = RotationMatrix(np.column_stack((x_axis, y_axis, z_axis)))
    return RigidTransform(R, camera_pos)

def build_single_arm_env():
    repo_dir = RepoDir()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(os.path.join(repo_dir, "package.xml"))
    parser.package_map().AddPackageXml(os.path.join(repo_dir, "models/universal_robots/ur_description/package.xml"))
    
    single_yaml = os.path.join(repo_dir, "models/ur5e_single_arm_on_table.yaml")
    yaml_content = """directives:
  - add_model:
      name: ur5e
      file: package://eaik_ift_experiment/models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf
  - add_weld:
      parent: world
      child: ur5e::base_link
  - add_model:
      name: wsg
      file: package://eaik_ift_experiment/models/wsg_finray/wsg50_110_finray_fingers_box_collision_obj.sdf
  - add_weld:
      parent: ur5e::tool0
      child: wsg::body_frame
      X_PC:
        translation: [0, 0, 0.04]
        rotation: !Rpy { deg: [90, 0, 0] }
  - add_model:
      name: table
      file: package://drake_models/manipulation_station/table_wide.sdf
  - add_weld:
      parent: world
      child: table::table_body
      X_PC:
        translation: [0.4, 0.0, 0.0]
  # table2 is the robot's standing surface.  It must be here as well as in the render
  # scene, or solutions are never collision-checked against something the figure shows.
  - add_model:
      name: table2
      file: package://drake_models/manipulation_station/table_wide.sdf
  - add_weld:
      parent: world
      child: table2::table_body
      X_PC:
        translation: [-0.2, 0.0, 0.0]
  - add_model:
      name: target_mug
      file: package://eaik_ift_experiment/models/mug/mug_simple_red.urdf
  - add_weld:
      parent: world
      child: target_mug::mug_body_link
      X_PC:
        translation: [0.38, 0.0, 0.06]
"""
    with open(single_yaml, "w") as f:
        f.write(yaml_content)
            
    ProcessModelDirectives(LoadModelDirectives(single_yaml), plant, parser)
    plant.Finalize()
    diagram = builder.Build()
    return diagram, plant, scene_graph

def main():
    parser = argparse.ArgumentParser(description="Visualize three UR5e robots grasping a mug with dropped opacity using Direct Reachability.")
    parser.add_argument("--seed", type=int, default=692332, help="Random seed for IK guesses")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the grippers (0.0 to 1.0).")
    parser.add_argument("--arm-alpha", type=float, default=0.4,
                        help="Opacity of the arm links. 1.0 keeps them solid, as in the "
                             "original figure; lower values let the mug read through them.")
    parser.add_argument("--html-out", type=str, default="out/grasp_selection.html",
                        help="Where to save the Meshcat scene for the Blender render pipeline.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Solve, export the scene, and exit without serving Meshcat.")
    parser.add_argument("--seed-search", type=int, default=0,
                        help="Try this many extra seeds and keep the one whose three grasps "
                             "are most spread around the mug.")
    parser.add_argument("--min-spread", type=float, default=60.0,
                        help="Stop the seed search once the minimum pairwise approach-angle "
                             "gap reaches this many degrees.")
    parser.add_argument("--roll-pitch-deg", type=float, default=40.0,
                        help="Bound on grasp roll/pitch in the mug frame, in degrees. Small "
                             "values force upright grasps that differ only in yaw.")
    parser.add_argument("--yaw-centers", type=float, nargs="+", default=[-140.0, 0.0, 140.0],
                        help="Yaw sector centres for the three grasps, in degrees.")
    parser.add_argument("--yaw-sector", type=float, default=40.0,
                        help="Width of each yaw sector, in degrees.")
    parser.add_argument("--configurations", type=str, default=None,
                        help="JSON file holding three pinned joint configurations to pose "
                             "directly, skipping the solve. This is how the published "
                             "figure is reproduced: it was solved by an older revision and "
                             "no seed reproduces it (see the file's own comment). Omit to "
                             "solve fresh grasps with the current solver.")
    args = parser.parse_args()
    repo_dir = RepoDir()
    
    generate_obj_description_files()
    
    print("Solving Optimization IK for 3 grasp configurations using Direct Reachability...")
    diagram_ik, plant_ik, scene_graph_ik = build_single_arm_env()
    diagram_ik_context = diagram_ik.CreateDefaultContext()
    plant_ik_context = plant_ik.GetMyContextFromRoot(diagram_ik_context)
    
    mug_instance = plant_ik.GetModelInstanceByName("target_mug")
    mug_frame = plant_ik.GetFrameByName("mug_body_link", mug_instance)
    X_WM = mug_frame.CalcPoseInWorld(plant_ik_context)
    
    problem = UrIKProblemNewFormulation(diagram_ik, use_boundary_reach=False)
    
    base_options = UrProblemOptions(
        solver="SNOPT",
        max_wall_time=10.0,
        avoid_collisions=True,
        impose_joint_centering_cost=True,
        joint_centering_cost_multiplier=5.0,
        target_mug=X_WM,
        ift_strategy="residual",
        ift_damping_lam=1e-3,
    )
    problem.ApplyOptions(base_options)
    

    def grasp_azimuths(q_solutions):
        """
        Approach azimuth of each grasp about the mug's axis, in degrees.

        The figure is only informative if the three grasps come from visibly different
        directions; two solutions can differ a lot in joint space and still put the
        gripper in the same place, so spread must be measured in the mug frame.
        """
        azimuths = []
        for q in q_solutions:
            plant_ik.SetPositions(plant_ik_context, problem.arm_instance, q)
            X_WG = problem.ee_frame.CalcPoseInWorld(plant_ik_context).multiply(problem.X_EG)
            approach = X_WM.inverse().multiply(X_WG).rotation().matrix()[:, 0]
            azimuths.append(np.degrees(np.arctan2(approach[1], approach[0])))
        return azimuths

    def azimuth_spread(q_solutions):
        """Smallest pairwise angular gap between grasps; larger is a better figure."""
        if len(q_solutions) < 3:
            return -1.0
        az = grasp_azimuths(q_solutions)
        return min(abs((az[a] - az[b] + 180) % 360 - 180)
                   for a in range(len(az)) for b in range(a + 1, len(az)))

    def solve_for_seed(current_seed):
        """
        Solve for three grasps that read clearly in a figure.

        Two figure-only devices, neither of which the benchmark uses:
          * roll and pitch of the grasp frame are pinned near zero, so every grasp is
            upright and the three differ essentially only in yaw about the mug axis;
          * each solve is restricted to its own yaw sector, so the three are separated by
            construction instead of by searching seeds for a lucky spread.
        """
        q_solutions = []
        yaw_centers = np.deg2rad(args.yaw_centers)
        half_sector = np.deg2rad(args.yaw_sector) / 2.0
        angles_to_try = [0.0, 2*np.pi/3, -2*np.pi/3, np.pi/2, -np.pi/2, np.pi,
                         np.pi/3, -np.pi/3, np.pi/4, -np.pi/4]

        for slot, yaw_center in enumerate(yaw_centers):
            solved = False
            # Seed each sector from whichever approach angle is closest to it.
            ordered = sorted(angles_to_try,
                             key=lambda t: abs((t - yaw_center + np.pi) % (2*np.pi) - np.pi))
            for i, theta_base in enumerate(ordered[:4]):
                q_init, theta, d_grasp, z_val, X_WE_init = get_initial_guess_for_angle(
                    plant_ik, plant_ik_context, scene_graph_ik, diagram_ik_context,
                    theta_base, current_seed + slot * 1000 + i * 100
                )

                # p is the grasp frame G in the mug frame M, not tool0 in M.
                p_init = problem.GraspParamsFromQ(q_init, X_WM)

                options = copy.copy(base_options)
                options.grasp_roll_pitch_bound = np.deg2rad(args.roll_pitch_deg)
                options.grasp_yaw_bounds = (yaw_center - half_sector,
                                            yaw_center + half_sector)
                # Centering on the seed rather than on the joint-limit midpoint is
                # deliberate here: it keeps each grasp near its own sector.  The benchmark
                # centers on the midpoint instead.  This must be set *before*
                # ApplyOptions, which snapshots q_nominal into the cost's q_target.
                problem.q_nominal = q_init
                problem.ApplyOptions(options)
                problem.q_branch_ref = q_init
                p_init[3:5] = np.clip(p_init[3:5], -options.grasp_roll_pitch_bound,
                                      options.grasp_roll_pitch_bound)
                p_init[5] = np.clip(p_init[5], *options.grasp_yaw_bounds)
                problem.prog.SetInitialGuess(problem.p, p_init)

                print(f"Solving yaw sector centred at {np.degrees(yaw_center):+.0f} deg "
                      f"(seed angle {theta_base:+.2f}) ...")
                res = problem.Solve()

                if res.is_success():
                    q_opt = problem.GetQ(res.get_x_val())
                    print(f"  Success! Optimal cost: {res.get_optimal_cost():.4f}")
                    q_solutions.append(q_opt)
                    solved = True
                    break
                print("  Solver failed; trying another seed angle for this sector.")
            if not solved:
                print(f"  No solution in the sector at {np.degrees(yaw_center):+.0f} deg.")
        return q_solutions

    print("Setting up render environment...")
    meshcat = Meshcat()
    meshcat.AddButton("Solve New Random Seed")
    builder_render = DiagramBuilder()
    plant_render, scene_graph_render = AddMultibodyPlantSceneGraph(builder_render, 0.01)
    

    parser_render = Parser(plant_render, scene_graph_render)
    parser_render.package_map().AddPackageXml(os.path.join(repo_dir, "package.xml"))
    parser_render.package_map().AddPackageXml(os.path.join(repo_dir, "models/universal_robots/ur_description/package.xml"))
    
    three_yaml = os.path.join(repo_dir, "models/ur5e_three_arms.yaml")
    yaml_content_3 = """directives:
  - add_model:
      name: table
      file: package://drake_models/manipulation_station/table_wide.sdf
  - add_weld:
      parent: world
      child: table::table_body
      X_PC:
        translation: [0.4, 0.0, 0.0]

  - add_model:
      name: table2
      file: package://drake_models/manipulation_station/table_wide.sdf
  - add_weld:
      parent: world
      child: table2::table_body
      X_PC:
        translation: [-0.2, 0.0, 0.0]

  - add_model:
      name: target_mug
      file: package://eaik_ift_experiment/models/mug/mug_simple_red.urdf
  - add_weld:
      parent: world
      child: target_mug::mug_body_link
      X_PC:
        translation: [0.38, 0.0, 0.06]
        rotation: !Rpy { deg: [0.0, 0.0, 0.0] }

  - add_model:
      name: ur5e_1
      file: package://eaik_ift_experiment/models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf
  - add_weld:
      parent: world
      child: ur5e_1::base_link
  - add_model:
      name: wsg_1
      file: package://eaik_ift_experiment/models/wsg_finray/wsg50_110_finray_fingers_box_collision_obj.sdf
  - add_weld:
      parent: ur5e_1::tool0
      child: wsg_1::body_frame
      X_PC:
        translation: [0, 0, 0.04]
        rotation: !Rpy { deg: [90, 0, 0] }

  - add_model:
      name: ur5e_2
      file: package://eaik_ift_experiment/models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf
  - add_weld:
      parent: world
      child: ur5e_2::base_link
  - add_model:
      name: wsg_2
      file: package://eaik_ift_experiment/models/wsg_finray/wsg50_110_finray_fingers_box_collision_obj.sdf
  - add_weld:
      parent: ur5e_2::tool0
      child: wsg_2::body_frame
      X_PC:
        translation: [0, 0, 0.04]
        rotation: !Rpy { deg: [90, 0, 0] }

  - add_model:
      name: ur5e_3
      file: package://eaik_ift_experiment/models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf
  - add_weld:
      parent: world
      child: ur5e_3::base_link
  - add_model:
      name: wsg_3
      file: package://eaik_ift_experiment/models/wsg_finray/wsg50_110_finray_fingers_box_collision_obj.sdf
  - add_weld:
      parent: ur5e_3::tool0
      child: wsg_3::body_frame
      X_PC:
        translation: [0, 0, 0.04]
        rotation: !Rpy { deg: [90, 0, 0] }
"""
    with open(three_yaml, "w") as f:
        f.write(yaml_content_3)
        
    ProcessModelDirectives(LoadModelDirectives(three_yaml), plant_render, parser_render)
    plant_render.Finalize()
    
    inspector = scene_graph_render.model_inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        properties = inspector.GetIllustrationProperties(geometry_id)
        if properties is not None:
            perc_properties = PerceptionProperties(properties)
            existing = inspector.GetPerceptionProperties(geometry_id)
            if existing is None:
                scene_graph_render.AssignRole(plant_render.get_source_id(), geometry_id, perc_properties, RoleAssign.kNew)
            else:
                scene_graph_render.RemoveRole(plant_render.get_source_id(), geometry_id, Role.kPerception)
                scene_graph_render.AssignRole(plant_render.get_source_id(), geometry_id, perc_properties)
                
    ur5e_models = ["ur5e_1", "ur5e_2", "ur5e_3", "wsg_1", "wsg_2", "wsg_3", "target_mug"]
    set_transparency_of_models(plant_render, ur5e_models, args.alpha, scene_graph_render,
                               arm_alpha=args.arm_alpha)
    

    target_pos = [0.38, 0.0, 0.12]
    camera_pos = [0.75, -0.45, 0.85]
    
    meshcat.SetCameraTarget(target_pos)
    meshcat.SetCameraPose(camera_pos, target_pos)
    
    MeshcatVisualizer.AddToBuilder(builder_render, scene_graph_render, meshcat)
    diagram_render = builder_render.Build()
    context = diagram_render.CreateDefaultContext()
    plant_context = plant_render.GetMyContextFromRoot(context)
    
    def pose_and_export(q_solutions):
        """Pose the three arms, publish to Meshcat, and save the scene for Blender."""
        for i, q in enumerate(q_solutions[:3], start=1):
            plant_render.SetPositions(
                plant_context, plant_render.GetModelInstanceByName(f"ur5e_{i}"), q
            )
        diagram_render.ForcedPublish(context)

        # The render scene contains table2, which the IK scene must also contain, or a
        # configuration could be shown resting inside geometry it was never checked
        # against.  Verify rather than assume.
        #
        # The three arms are three configurations of the SAME robot drawn together, so
        # they legitimately overlap each other; only each arm against the static scene
        # (and itself) is a real collision.
        query = scene_graph_render.get_query_output_port().Eval(
            scene_graph_render.GetMyContextFromRoot(context)
        )
        inspector = query.inspector()

        def arm_index(geometry_id):
            name = inspector.GetName(inspector.GetFrameId(geometry_id))
            for i in (1, 2, 3):
                if name.startswith(f"ur5e_{i}::") or name.startswith(f"wsg_{i}::"):
                    return i
            return 0  # static scene: tables and the mug

        min_dist = 1.0
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(0.01):
            a, b = arm_index(pair.id_A), arm_index(pair.id_B)
            if a != b and a != 0 and b != 0:
                continue  # different arms, expected to interpenetrate
            min_dist = min(min_dist, pair.distance)
        status = "OK" if min_dist > 0.0 else "IN COLLISION"
        print(f"Minimum signed distance, each arm vs the static scene: "
              f"{min_dist:+.4f} m ({status})")

        os.makedirs(os.path.dirname(os.path.abspath(args.html_out)), exist_ok=True)
        with open(args.html_out, "w") as f:
            f.write(meshcat.StaticHtml())
        print(f"Saved Meshcat scene to {args.html_out}")
        print("  Render it with: ./scripts/render_grasp_figure.sh")

    print(f"Meshcat URL: {meshcat.web_url()}")
    print("Press 'Solve New Random Seed' in the Meshcat GUI to generate a new grasp!")

    current_seed = args.seed
    click_count = 0

    if args.configurations:
        with open(args.configurations) as f:
            pinned = json.load(f)["configurations"]
        q_solutions = [np.asarray(q, dtype=float) for q in pinned]
        if len(q_solutions) < 3 or any(q.shape != (6,) for q in q_solutions):
            raise ValueError(
                f"{args.configurations} must hold at least three 6-element configurations")
        print(f"Posing {len(q_solutions)} pinned configurations from "
              f"{args.configurations} (no solve).")
    elif args.seed_search > 0:
        # Search for a seed whose three grasps approach from distinct directions.  The
        # previously hard-coded seed was picked under the old solver; branch selection
        # and the initial-guess mapping have both changed since, so a seed that once
        # gave a spread trio no longer does.
        rng = np.random.default_rng(args.seed)
        candidates = [args.seed] + [int(rng.integers(0, 1_000_000))
                                    for _ in range(args.seed_search)]
        best = (None, None, -1.0)
        for candidate in candidates:
            print(f"\n--- Trying seed {candidate} ---")
            sols = solve_for_seed(candidate)
            spread = azimuth_spread(sols)
            print(f"  seed {candidate}: {len(sols)} solutions, "
                  f"minimum azimuth gap {spread:.1f} deg")
            if spread > best[2]:
                best = (candidate, sols, spread)
            if spread >= args.min_spread:
                print(f"  accepted (>= {args.min_spread} deg)")
                break
        current_seed, q_solutions, spread = best
        print(f"\nBest seed: {current_seed} with minimum azimuth gap {spread:.1f} deg")
    else:
        q_solutions = solve_for_seed(current_seed)

    if len(q_solutions) >= 3:
        pose_and_export(q_solutions)
        if args.configurations:
            print(f"Success! Displaying the pinned configurations in Meshcat.")
        else:
            print(f"Success! Displaying seed {current_seed} in Meshcat.")
            print(f"USE THIS SEED LATER: {current_seed}")
        print(f"  grasp azimuths in the mug frame: "
              f"{[round(a, 1) for a in grasp_azimuths(q_solutions)]} deg")
    else:
        print(f"Failed to find 3 solutions for seed {current_seed}.")

    if args.no_interactive:
        return

    while True:
        new_clicks = meshcat.GetButtonClicks("Solve New Random Seed")
        if new_clicks > click_count:
            current_seed = np.random.randint(0, 1000000)
            print(f"\n--- Solving for seed {current_seed} ---")
            
            q_solutions = solve_for_seed(current_seed)
            if len(q_solutions) >= 3:
                pose_and_export(q_solutions)
                print(f"Success! Displaying seed {current_seed} in Meshcat.")
                print(f"USE THIS SEED LATER: {current_seed}")
            else:
                print(f"Failed to find 3 solutions for seed {current_seed}. Please click again.")
            
            click_count = new_clicks
            
        time.sleep(0.1)

if __name__ == "__main__":
    main()
