import os
from pydrake.all import(
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    ApplyVisualizationConfig,
    VisualizationConfig,
)

def RepoDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def BuildEnv(meshcat=None, directives_file=None, visualize=True):
    if directives_file is None:
        directives_file = os.path.join(RepoDir(), 'models/ur5e_collision.yaml')
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)

    # Load the model directives from the YAML file.
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    # Register the ur_description sub-package for UR URDF mesh resolution
    ur_description_xml = os.path.join(RepoDir(), "models/universal_robots/ur_description/package.xml")
    if os.path.exists(ur_description_xml):
        parser.package_map().AddPackageXml(ur_description_xml)
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)

    plant.Finalize()
    if visualize and meshcat is not None:
        vis_config = VisualizationConfig()
        vis_config.publish_illustration = True
        vis_config.publish_proximity = True
        vis_config.publish_inertia = True
        vis_config.delete_on_initialization_event = True   # Clear old visualizations
        ApplyVisualizationConfig(vis_config, builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram
