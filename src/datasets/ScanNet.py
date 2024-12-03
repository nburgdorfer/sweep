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

class ScanNet(BaseDataset):
    def __init__(self, cfg, mode, scenes):
        super(ScanNet, self).__init__(cfg, mode, scenes)
        self.png_depth_scale = 1000.0

    def get_frame_count(self, scene):
        image_files = os.listdir(os.path.join(self.data_path, scene, "color"))
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
                image_files.append(os.path.join(self.data_path, scene, "color", f"{ind:06d}.jpg"))
                depth_files.append(os.path.join(self.data_path, scene, "depth", f"{ind:06d}.png"))
                pose_files.append(os.path.join(self.data_path, scene, "pose", f"{ind:06d}.txt"))
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
        intrinsics_file = os.path.join(self.data_path, scene, "intrinsic", "intrinsic_depth.txt")
        K = np.loadtxt(intrinsics_file).astype('float32')
        self.K[scene] = K[:3,:3]
        self.H = int(self.scale * (self.cfg["camera"]["height"] - self.crop_h))
        self.W = int(self.scale * (self.cfg["camera"]["width"]- self.crop_w))
        self.K[scene] = scale_cam(self.K[scene], scale=self.scale)

    def get_pose(self, pose_file, frame_id=None):
        pose = np.loadtxt(pose_file).astype('float32')
        pose = np.linalg.inv(pose)
        if(np.isnan(pose).any()):
            print(pose, pose_file)
        return pose

    def get_all_poses(self, scene):
        poses = []
        pose_path = os.path.join(self.data_path, scene, "pose")
        pose_files = os.listdir(pose_path)
        pose_files.sort()
        for pose_file in pose_files:
            pose = np.loadtxt(os.path.join(pose_path, pose_file)).astype('float32')
            if self.mode == "inference":
                pose = np.linalg.inv(pose)
            if(np.isnan(pose).any()):
                print(pose, pose_file)
            poses.append(pose)
        return poses
