import argparse
import os
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import shutil
import sys
from tqdm import tqdm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

from cvt.io import read_cluster_list, write_pfm

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4 # already downsampled
    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    all_views = list(range(0,49))
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            data.append((ref_view, src_views))
    return data

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, pix_th=1.0):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = dist < pix_th
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

def laplacian_filter(cfg, est_depth_path, est_conf_path, rgb_path, laplacian_path, output_path, dataset, scene):
    K = dataset.K[scene]
    poses = dataset.get_all_poses(scene)
    prob_th = cfg["point_cloud"]["prob_th"]
    image_region_th = cfg["point_cloud"]["image_region_th"]
    depth_region_th = cfg["point_cloud"]["depth_region_th"]

    vertexs = []
    vertex_colors = []

    clusters = read_cluster_list(dataset.get_cluster_file(scene))
    nviews = len(clusters)

    out_mask_path = os.path.join(output_path, "masks")
    lap_mask_path = os.path.join(out_mask_path, "laplacian")
    conf_mask_path = os.path.join(out_mask_path, "confidence")
    os.makedirs(out_mask_path, exist_ok=True)
    os.makedirs(lap_mask_path, exist_ok=True)
    os.makedirs(conf_mask_path, exist_ok=True)

    # for each reference view and the corresponding source views
    for (ref_frame, _) in tqdm(clusters, desc="Building point cloud", unit="views"):
        ref_image = cv2.imread(os.path.join(rgb_path, f"{ref_frame:08d}.png"))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB ) 
        ref_depth_est = read_pfm(os.path.join(est_depth_path, f"{ref_frame:08d}.pfm"))[0]
        ref_pose = poses[ref_frame]

        #   ### Confidence Mask ###
        #   confidence = read_pfm(os.path.join(est_conf_path, f"{ref_frame:08d}.pfm"))[0]
        #   conf_mask = confidence > prob_th
        #   if conf_mask.shape[0] != ref_depth_est.shape[0] or conf_mask.shape[1] != ref_depth_est.shape[1]:
        #       conf_mask_t = torch.tensor(conf_mask, dtype=torch.float32)
        #       conf_mask = torch.squeeze(
        #           F.interpolate(
        #               torch.unsqueeze(torch.unsqueeze(conf_mask_t, 0), 0), 
        #               [ref_depth_est.shape[0], ref_depth_est.shape[1]], mode="nearest")).numpy() == 1.0

        ### Laplacian Mask ###
        image_lap = read_pfm(os.path.join(laplacian_path, f"{ref_frame:08d}_image.pfm"))[0]
        image_lap_mask = np.where(image_lap >= image_region_th, 1, 0)
        depth_lap = read_pfm(os.path.join(laplacian_path, f"{ref_frame:08d}_depth.pfm"))[0]
        depth_lap_mask = np.where(depth_lap <= depth_region_th, 1, 0)
        laplacian_mask = (image_lap_mask * depth_lap_mask).astype(bool)

        full_mask = laplacian_mask
        #full_mask = laplacian_mask * conf_mask

        #save_mask(os.path.join(conf_mask_path, f"{ref_frame:08d}_conf.png"), conf_mask)
        save_mask(os.path.join(lap_mask_path, f"{ref_frame:08d}_lap.png"), laplacian_mask)
        save_mask(os.path.join(out_mask_path, f"{ref_frame:08d}_full.png"), full_mask)

        height, width = ref_depth_est.shape
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        # save filtered depth map
        filtered_depth = ref_depth_est * full_mask
        depth_filename = os.path.join(est_depth_path, f"{ref_frame:08d}_filt.pfm")
        write_pfm(depth_filename, filtered_depth)
        filtered_depth = (filtered_depth - cfg["camera"]["near"]) / (cfg["camera"]["far"]-cfg["camera"]["near"])
        os.makedirs(os.path.join(est_depth_path, "disp"), exist_ok=True)
        depth_filename = os.path.join(est_depth_path, "disp", f"{ref_frame:08d}_filt.png")
        cv2.imwrite(depth_filename, (filtered_depth * 255))


        # use either average or reference depth estimates
        x, y, depth = x[laplacian_mask], y[laplacian_mask], ref_depth_est[laplacian_mask]

        color = ref_image[laplacian_mask]
        xyz_ref = np.matmul(np.linalg.inv(K),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_pose),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color[:,:3]).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')

    # save point cloud to scene-level points path
    ply_file_scene = os.path.join(output_path, f"{scene}.ply")
    PlyData([el]).write(ply_file_scene)
