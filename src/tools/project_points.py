import os
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d

from cvt.io import read_pfm
from cvt.geometry import _points_from_depth

def project_points(cfg, est_depth_path, est_conf_path, rgb_path, output_path, dataset, scene, voxel_size=0.03, confidence_th=0.75):
    K = dataset.K[scene]
    poses = dataset.get_all_poses(scene)

    points = []
    colors = []
    confidence = []
    view_inds = []

    # for each reference view and the corresponding source views
    image_files = os.listdir(rgb_path)
    image_files.sort()
    for image_file in tqdm(image_files, desc="Building point cloud", unit="view"):
        view_num = int(image_file[:8])
        image = cv2.imread(os.path.join(rgb_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        est_depth = read_pfm(os.path.join(est_depth_path, f"{view_num:08d}.pfm"))
        est_conf = read_pfm(os.path.join(est_conf_path, f"{view_num:08d}.pfm"))
        pose = poses[view_num]
        
        est_points, valid_inds = _points_from_depth(est_depth, K, pose)
        est_points = est_points[valid_inds]
        image = image.reshape(-1,3)[valid_inds]
        est_conf = est_conf.flatten()[valid_inds]
        view = np.ones((est_points.shape[0])) * view_num

        ## confidence filtering
        conf_inds = np.where(est_conf >= confidence_th)[0]
        est_points = est_points[conf_inds]
        image = image[conf_inds]
        est_conf = est_conf[conf_inds]
        view = view[conf_inds]
        
        points.append(est_points)
        colors.append(image)
        confidence.append(est_conf)
        view_inds.append(view)

    points = np.concatenate(points, axis=0).astype(np.float32)
    colors = np.concatenate(colors, axis=0).astype(np.float32) / 255
    confidence = np.concatenate(confidence, axis=0).astype(np.float32)
    view_inds = np.concatenate(view_inds, axis=0).astype(np.float32)

    os.makedirs(output_path, exist_ok=True)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(os.path.join(output_path, f"{scene}.ply"), cloud)

    #cloud = {
    #        "points": points,
    #        #"colors": colors,
    #        "confidence": confidence,
    #        "view_inds": view_inds
    #        }
    #np.save(os.path.join(output_path, f"{scene}.npy"), cloud)
