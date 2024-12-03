import sys
import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import imageio
import yaml

import open3d as o3d
import matplotlib.pyplot as plt

TOOLS_ROOT=os.path.abspath(os.path.dirname(__file__))
SRC_ROOT=os.path.abspath(os.path.dirname(TOOLS_ROOT))
TOP_ROOT=os.path.abspath(os.path.dirname(SRC_ROOT))
sys.path.append(TOP_ROOT)

from src.utils.camera import build_o3d_traj
from src.utils.io import read_mesh, read_cams_traj, read_single_cam_sfm
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

def load_bounding_box(bb_path):
    with open(bb_path, 'r') as f:
        bb_config = yaml.full_load(f)
    return bb_config

def main():
    cam_pyrs_path = sys.argv[1]
    mesh_path = sys.argv[2]
    scene_bound_file = sys.argv[3]
    cam_file = sys.argv[4]

    # read scene bound info
    bb = load_bounding_box(scene_bound_file)
    bound = np.asarray(bb["mapping"]["bound"])
    center_point = bound.mean(axis=1)
    max_dist = np.max(np.abs(bound[:,1]-bound[:,0]))

    # get example camera
    cam = read_single_cam_sfm(cam_file)
    P = cam[0]
    K = cam[1]
    print(P)
    R = P[:3,:3]
    t = P[:3,3]
    C = -R.transpose() @ t
    z = R[2,:]
    print(R)
    print(t)
    print(C)
    print(z)

    new_optical_axis = center_point-C
    print(new_optical_axis)


if __name__=="__main__":
    main()


