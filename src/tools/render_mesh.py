import sys
import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import imageio

import open3d as o3d
import matplotlib.pyplot as plt

TOOLS_ROOT=os.path.abspath(os.path.dirname(__file__))
SRC_ROOT=os.path.abspath(os.path.dirname(TOOLS_ROOT))
TOP_ROOT=os.path.abspath(os.path.dirname(SRC_ROOT))
sys.path.append(TOP_ROOT)

from src.utils.camera import build_o3d_traj
from src.utils.io import read_mesh, read_cams_traj
from src.datasets.BaseDataset import build_dataset
from src import config
from src.utils.common import as_intrinsics_matrix


def render(mesh_files, trajectory, output_path, dataset, frames):
    render.index = -1
    render.mesh_index = 0
    render.trajectory=trajectory
    render.vis = o3d.visualization.Visualizer()

    num_meshes = len(mesh_files)
    mesh_offset = frames//num_meshes
    mesh = read_mesh(mesh_files[0])
    mesh.compute_vertex_normals()

    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = render
        if (glb.index >= 0):
            if ((glb.index % mesh_offset == 0) and (glb.index != 0) and (glb.mesh_index < num_meshes-1)):
                glb.mesh_index += 1
                mesh = read_mesh(mesh_files[glb.mesh_index])
                mesh.compute_vertex_normals()
                vis.clear_geometries()
                vis.add_geometry(mesh)


            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("./{:05d}.png".format(glb.index), False)
            # depth = vis.capture_depth_float_buffer(False)
            # plt.imsave(os.path.join(depth_path, "{:05d}.png".format(glb.index)), np.asarray(depth), dpi = 1)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(output_path, "{:05d}.png".format(glb.index)), np.asarray(image), dpi = 1)
        glb.index = glb.index + 1

        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index], True)
        else:
            render.vis.register_animation_callback(None)
            vis.destroy_window()
        return False
    
    vis = render.vis
    vis.create_window(width=dataset.W, height=dataset.H)
    vis.add_geometry(mesh)
    vis.get_render_option().load_from_json("src/tools/render_options.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    return

def render_mesh(cfg, poses, dataset, scene, render_path, mesh=None, mesh_path=None, frames=-1, invert=False):
    #if not o3d._build_config['ENABLE_HEADLESS_RENDERING']:
    #    print("Headless rendering is not enabled. "
    #          "Please rebuild Open3D with ENABLE_HEADLESS_RENDERING=ON")
    #    sys.exit()

    if frames != -1:
        poses = poses[:frames]
    else:
        frames = len(poses)

    # convert poses to open3d trajectory list format
    trajectory = build_o3d_traj(poses, dataset.K[scene], width=dataset.W, height=dataset.H)

    # create list of mesh files
    if ((mesh == None) and (mesh_path != None)):
        mesh_files = os.listdir(mesh_path)
        mesh_files.sort()
        mesh_files = [ os.path.join(mesh_path,mf) for mf in mesh_files if ".ply" in mf]
    else:
        mesh_files = [mesh]

    # create output directory
    output_path = os.path.join(cfg["output_path"], scene, render_path)
    os.makedirs(output_path, exist_ok=True)

    # render mesh into camera frames
    render(mesh_files, trajectory, output_path, dataset, frames)
