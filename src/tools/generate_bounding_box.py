import numpy as np
import os
import sys
import open3d as o3d

def read_single_cam_sfm(cam_file: str, depth_planes: int = 256) -> np.ndarray:
    cam = np.zeros((2, 4, 4))

    with open(cam_file, 'r') as cam_file:
        words = cam_file.read().split()

    words_len = len(words)

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0,i,j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1,i,j] = float(words[intrinsic_index])

    if words_len == 29:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = depth_planes
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 30:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 31:
        cam[1,3,0] = words[27]
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = float(words[30])
    else:
        cam[1,3,0] = 0
        cam[1,3,1] = 0
        cam[1,3,2] = 0
        cam[1,3,3] = 1

    return cam

def dtu(scene):
    pose_path = f"/home/s9053071/data/DTU/{scene}/cams"
    pose_files = os.listdir(pose_path)
    pose_files.sort()
    pose_files = [ os.path.join(pose_path, pf) for pf in pose_files if pf[-8:]=="_cam.txt"]
    mesh = o3d.io.read_triangle_mesh(f"/home/s9053071/data/DTU/{scene}/{scene}.ply")
    for pf in pose_files:
        cam = read_single_cam_sfm(pf)
        pose = cam[0,:,:]
        R = pose[:3,:3]
        t = pose[:3,3]
        C = -R.transpose() @ t
        mesh.vertices.append(t)
    return mesh

def tnt(scene):
    pose_path = f"/home/s9053071/data/TNT/{scene}/Cameras"
    pose_files = os.listdir(pose_path)
    pose_files.sort()
    pose_files = [ os.path.join(pose_path, pf) for pf in pose_files if pf[-8:]=="_cam.txt"]
    mesh = o3d.io.read_triangle_mesh(f"/home/s9053071/data/TNT/{scene}/{scene}.ply")
    for pf in pose_files:
        cam = read_single_cam_sfm(pf)
        pose = cam[0,:,:]
        R = pose[:3,:3]
        t = pose[:3,3]
        C = -R.transpose() @ t
        mesh.vertices.append(t)
    return mesh

def main():

    if (len(sys.argv) == 3):
        scale = 1.0
    elif (len(sys.argv) == 4):
        scale = float(sys.argv[3])
    else:
        print(f"Error! usage: python {sys.argv[0]} <dataset> <scene> [scale]")
        sys.exit()

    dataset = sys.argv[1]
    scene = sys.argv[2]

    if dataset=="dtu":
        mesh = dtu(scene)
    elif dataset=="tnt":
        mesh = tnt(scene)
    else:
        print(f"Error! Unknown dataset {dataset}.")

    bounding = mesh.get_axis_aligned_bounding_box()
    min_bound = bounding.get_min_bound()/scale # divide by 1000 to convert to meters
    max_bound = bounding.get_max_bound()/scale # divide by 1000 to convert to meters
    buffer_pad_percent = 0.05
    bound_range = max_bound-min_bound
    pad = bound_range * buffer_pad_percent
    min_bound -= pad
    max_bound += pad
    print(f"Min Bound: {min_bound}")
    print(f"Max Bound: {max_bound}")


    print(f"[[{min_bound[0]:0.3f}, {max_bound[0]:0.3f}], [{min_bound[1]:0.3f}, {max_bound[1]:0.3f}], [{min_bound[2]:0.3f}, {max_bound[2]:0.3f}]]")

if __name__=="__main__":
    main()
