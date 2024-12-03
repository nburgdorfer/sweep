import os
import random
import numpy as np
import torch
import cv2
import sys
import json
from tqdm import tqdm
import yaml

from src.utils.camera import scale_cam
from src.utils.io import read_single_cam_sfm, read_pfm
from src.datasets.BaseDataset import BaseDataset

class Replica(BaseDataset):
    def __init__(self, cfg, mode, scenes):
        super(Replica, self).__init__(cfg, mode, scenes)
        self.png_depth_scale = 6553.5

    def get_frame_count(self, scene):
        image_files = os.listdir(os.path.join(self.data_path, scene, 'images'))
        image_files = [img for img in image_files if img[-4:]==".jpg"]
        return len(image_files)

    def build_samples(self):
        self.total_samples = []
        self.frame_count = 0

        if self.mode=="inference":
            scenes = self.scenes
        else:
            scenes = tqdm(self.scenes, desc="Loading scene paths", unit="scene")

        for scene in scenes:
            # build samples dict for other data
            curr_frame_count = self.get_frame_count(scene)
            self.total_samples.extend(self.build_samples_helper(scene, curr_frame_count))
            self.frame_count += curr_frame_count

            # load pose and intrinsics for current scene
            self.load_intrinsics(scene)

        self.total_samples = np.asarray(self.total_samples)
        return


    def build_samples_helper(self, scene, frame_count):
        samples = []

        frame_offset = ((self.num_frame-1)//2)
        radius = frame_offset*self.frame_spacing
        for ref_frame in range(frame_count):
            skip = False
            if ((ref_frame + radius >= frame_count) or (ref_frame - radius < 0)):
                continue
            start = ref_frame - radius
            end = ref_frame + radius

            frame_inds = [i for i in range(ref_frame, end+1, self.frame_spacing) if i != ref_frame]
            bottom = [i for i in range(ref_frame, start-1, -self.frame_spacing) if i != ref_frame]
            frame_inds.extend(bottom)
            frame_inds.insert(0, ref_frame)

            image_files = []
            depth_files = []
            pose_files = []
            for ind in frame_inds:
                image_files.append(os.path.join(self.data_path, scene, "images", f"frame{ind:06d}.jpg"))
                depth_files.append(os.path.join(self.data_path, scene, "gt_depths", f"depth{ind:06d}.png"))
                pose_files.append(os.path.join(self.data_path, scene, "poses.txt"))
                if not os.path.isfile(pose_files[-1]):
                    #print(f"{pose_files[-1]} does not exist; skipping")
                    skip = True
            if skip:
                continue

            samples.append({"scene": scene,
                            "frame_inds": frame_inds,
                            "image_files": image_files,
                            "depth_files": depth_files,
                            "pose_files": pose_files
                            })
        return samples

    def load_intrinsics(self, scene):
        intrinsics_file = os.path.join(self.data_path, scene, "intrinsics.json")
        K = np.zeros((3,3), dtype=np.float32)
        with open(intrinsics_file) as f:
            data = json.load(f)
        K[0,0] = data["camera"]["fx"]
        K[1,1]= data["camera"]["fy"]
        K[0,2]= data["camera"]["cx"]
        K[1,2]= data["camera"]["cy"]
        K[2,2] = 1
        self.K[scene] = K
        self.H = int(self.scale * (self.cfg["camera"]["height"] - self.crop_h))
        self.W = int(self.scale * (self.cfg["camera"]["width"]- self.crop_w))
        self.K[scene] = scale_cam(self.K[scene], scale=self.scale)

    def get_pose(self, pose_file, frame_id=None):
        with open(pose_file, "r") as f:
            lines = f.readlines()
        pose = np.array(list(map(float, lines[frame_id].split()))).reshape(4, 4)
        pose = np.linalg.inv(pose)
        return pose

    def get_all_poses(self, scene):
        pose_file = os.path.join(self.data_path, scene, "poses.txt")
        poses = []
        with open(pose_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            poses.append(pose)
        return np.asarray(poses)
