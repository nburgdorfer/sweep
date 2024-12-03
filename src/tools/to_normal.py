import open3d as o3d
import numpy as np
import sys, os

def to_normal(ply_file, output_file):
    cloud = o3d.io.read_point_cloud(ply_file)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=50))
    normals = np.asarray(cloud.normals)
    normals = (normals-normals.min(axis=0)) / (normals.max(axis=0)-normals.min(axis=0))
    cloud.colors = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(output_file, cloud)
