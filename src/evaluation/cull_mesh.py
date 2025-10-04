import os
import argparse
import glob

import numpy as np
import torch
import imageio
import trimesh
import pyrender
from tqdm import tqdm
from copy import deepcopy

from src.utils.camera import to_opengl_pose


def load_virt_cam_poses(path):
    poses = []
    pose_paths = sorted(
        glob.glob(os.path.join(path, "*.txt")),
        key=lambda x: int(os.path.basename(x)[:-4]),
    )
    for pose_path in pose_paths:
        with open(pose_path, "r") as f:
            lines = f.readlines()
        ls = []
        for line in lines:
            l = list(map(float, line.split(" ")))
            ls.append(l)
        c2w = np.array(ls).reshape(4, 4)
        poses.append(np.linalg.inv(c2w))

    assert len(poses) > 0, "Make sure the path: {} really has virtual views!".format(
        path
    )
    print("Added {} virtual views from {}".format(len(poses), path))

    return poses


def load_cam_intrinsics_from_cfg(cfg):
    K = np.array(
        [
            [cfg["cam"]["fx"], 0.0, cfg["cam"]["cx"]],
            [0.0, cfg["cam"]["fy"], cfg["cam"]["cy"]],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3)
    H, W = cfg["cam"]["H"], cfg["cam"]["W"]
    return K, H, W


def render_depth_maps(mesh, poses, K, H, W, near=0.01, far=5.0):
    """
    :param mesh: Mesh to be rendered
    :param poses: list of camera poses (c2w under OpenGL convention)
    :param K: camera intrinsics [3, 3]
    :param W: width of image plane
    :param H: height of image plane
    :param near: near clip
    :param far: far clip
    :return: list of rendered depth images [H, W]
    """
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=near, zfar=far
    )
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        pose = to_opengl_pose(pose)
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps


def render_depth_maps_doublesided(mesh, poses, K, H, W, near=0.01, far=10.0):
    depth_maps_1 = render_depth_maps(mesh, poses, K, H, W, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, K, H, W, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[
        :, [2, 1]
    ]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in tqdm(range(len(depth_maps_1))):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where(
            (depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map),
            depth_maps_2[i],
            depth_map,
        )
        depth_maps.append(depth_map)

    return depth_maps


def cull_by_bounds(points, scene_bounds, epsilon=0.02):
    """
    :param points:
    :param scene_bounds:
    :param epsilon:
    :return:
    """
    inside_mask = np.all(points >= (scene_bounds[0] - epsilon), axis=1) & np.all(
        points <= (scene_bounds[1] + epsilon), axis=1
    )
    return inside_mask


def cull_from_one_pose(
    points, pose, K, H, W, remove_occlusion=True, rendered_depth=None, epsilon=0.03
):
    """
    :param points: mesh vertices [V, 3] np array
    :param pose: c2w under OpenGL convention (right-up-back) [3, 3] np array
    :param K: camera intrinsics [3, 3] np array
    :param rendered_depth: rendered depth image (optional)
    :param remove_occlusion:
    :return:
    """
    c2w = deepcopy(pose)
    # convert to OpenCV pose
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera coordinate frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: frustum
    in_frustum_mask = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)

    # step 2: self occlusion
    obs_mask = in_frustum_mask
    if remove_occlusion:
        assert (
            rendered_depth is not None
        ), "remove_occlusion requires rendered depth image!!!"
        obs_mask = in_frustum_mask & (pz < (rendered_depth[v, u] + epsilon))

    return in_frustum_mask.astype(np.int32), obs_mask.astype(np.int32)


def apply_culling_strategy(
    points,
    poses,
    K,
    H,
    W,
    rendered_depth_list=None,
    remove_occlusion=True,
    verbose=False,
    virt_cam_starts=-1,
    epsilon=0.03,
):
    """
    :param points:
    :param poses:
    :param K:
    :param H:
    :param W:
    :param rendered_depth_list:
    :param remove_occlusion:
    :param verbose:
    :param virt_cam_starts:
    :return:
    """

    in_frustum_mask = np.zeros(points.shape[0])
    obs_mask = np.zeros(points.shape[0])
    for i, pose in enumerate(tqdm(poses)):
        if verbose:
            print("Processing pose " + str(i + 1) + " out of " + str(len(poses)))
        rendered_depth = (
            rendered_depth_list[i] if rendered_depth_list is not None else None
        )
        in_frustum, obs = cull_from_one_pose(
            points,
            pose,
            K,
            H,
            W,
            rendered_depth=rendered_depth,
            remove_occlusion=remove_occlusion,
            epsilon=epsilon,
        )
        obs_mask = obs_mask + obs
        # virtual camera views shouldn't contribute to in_frustum_mask, it only adds more entries to obs_mask
        if virt_cam_starts < 0 or i < virt_cam_starts:
            in_frustum_mask = in_frustum_mask + in_frustum

    return in_frustum_mask, obs_mask


def cull_one_mesh(
    cfg, poses, K, mesh_file, culled_mesh_file, bounds=None, virt_cam_path=None
):
    # load settings
    epsilon = cfg["mesh_culling"]["epsilon"]
    max_edge = cfg["mesh_culling"]["max_edge"]
    platform = cfg["mesh_culling"]["platform"]
    remove_occlusion = cfg["mesh_culling"]["remove_occlusion"]
    save_unseen = cfg["mesh_culling"]["save_unseen"]
    scene_bounds = cfg["mesh_culling"]["scene_bounds"]
    silent = cfg["mesh_culling"]["silent"]
    subdivide = cfg["mesh_culling"]["subdivide"]
    th_obs = cfg["mesh_culling"]["th_obs"]
    virtual_cameras = cfg["mesh_culling"]["virtual_cameras"]
    H, W = cfg["camera"]["height"], cfg["camera"]["width"]

    # load original mesh
    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    vertices = mesh.vertices  # [V, 3]
    triangles = mesh.faces  # [F, 3]
    colors = mesh.visual.vertex_colors  # [V, 3]

    # cull with the bounding box first
    if scene_bounds:
        inside_mask = cull_by_bounds(vertices, bounds)
        inside_mask = (
            inside_mask[triangles[:, 0]]
            | inside_mask[triangles[:, 1]]
            | inside_mask[triangles[:, 2]]
        )
        triangles = triangles[inside_mask, :]

    os.environ["PYOPENGL_PLATFORM"] = platform

    # add virtual cameras to camera poses list
    # virt_cam_starts = -1
    if virtual_cameras:
        virt_cam_starts = len(poses)
        if virt_cam_path is None:
            virt_cam_path = os.path.join(
                cfg["data_path"], "virtual_cameras"
            )  # hardcoded path
        poses = poses + load_virt_cam_poses(virt_cam_path)
    else:
        virt_cam_starts = -1

    # we don't need to subdivided mesh to render depth, so do the rendering before updating the mesh
    if remove_occlusion:
        rendered_depth_maps = render_depth_maps_doublesided(
            mesh, poses, K, H, W, near=0.01, far=10.0
        )
    else:
        rendered_depth_maps = None

    # update the mesh vertices and faces
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # start culling
    points = vertices[:, :3]  # [V, 3]
    in_frustum_mask, obs_mask = apply_culling_strategy(
        points,
        poses,
        K,
        H,
        W,
        rendered_depth_list=rendered_depth_maps,
        verbose=(not silent),
        remove_occlusion=remove_occlusion,
        virt_cam_starts=virt_cam_starts,
        epsilon=epsilon,
    )
    inf1 = in_frustum_mask[triangles[:, 0]]  # [F, 3]
    inf2 = in_frustum_mask[triangles[:, 1]]
    inf3 = in_frustum_mask[triangles[:, 2]]
    in_frustum_mask = (inf1 > th_obs) | (inf2 > th_obs) | (inf3 > th_obs)
    if remove_occlusion:
        obs1 = obs_mask[triangles[:, 0]]
        obs2 = obs_mask[triangles[:, 1]]
        obs3 = obs_mask[triangles[:, 2]]
        obs_mask = (obs1 > th_obs) | (obs2 > th_obs) | (obs3 > th_obs)
        valid_mask = in_frustum_mask & obs_mask  # [F,]
    else:
        valid_mask = in_frustum_mask
    triangles_observed = triangles[valid_mask, :]

    # save culled mesh
    mesh = trimesh.Trimesh(
        vertices, triangles_observed, vertex_colors=colors, process=False
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(culled_mesh_file)

    # save unobserved points
    if save_unseen:
        triangles_not_observed = triangles[~valid_mask, :]
        mesh_not_observed = trimesh.Trimesh(
            vertices, triangles_not_observed, process=False
        )
        mesh_not_observed.remove_unreferenced_vertices()

        unseen_mesh_file = culled_mesh_file.replace(".ply", "_unseen.ply")
        mesh_not_observed.export(unseen_mesh_file)
        # pc_unseen = mesh_not_observed.vertices
        # np.save("{}/{}_pc_unseen.npy".format(save_dir, scene_name), pc_unseen)
