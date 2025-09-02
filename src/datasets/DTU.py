import os
import numpy as np
import sys
from tqdm import tqdm

from typing import Any
from numpy.typing import NDArray

from cvtkit.io import read_single_cam_sfm, read_cluster_list

from src.datasets.BaseDataset import BaseDataset

class DTU(BaseDataset):
    def __init__(self, cfg: dict[Any,Any], mode: str, scenes: list[str]):
        super(DTU, self).__init__(cfg, mode, scenes)
        self.units = "mm"
        if mode=="inference":
            self.gt_depth_path = os.path.join(self.data_path, "Depths", scenes[0])

    def get_frame_count(self, scene: str):
        image_files = os.listdir(os.path.join(self.data_path, "Images", "Lighting", scene))
        image_files = [img for img in image_files if img[-4:]==".png"]
        return len(image_files)

    def build_samples(self) -> tuple[list[dict[str,Any]], int]:
        total_samples: list[dict[str,Any]] = []
        frame_count = 0

        if self.mode=="inference":
            scenes = self.scenes
        else:
            scenes = tqdm(self.scenes, desc="Loading scene paths", unit="scene")

        for scene in scenes:
            if not os.path.isdir(os.path.join(self.data_path, "Images", scene)):
                print(f"{os.path.join(self.data_path, scene)} is not a valid directory.")
                continue

            # build samples dict for other data
            curr_frame_count = self.get_frame_count(scene)
            if self.sample_mode=="stream":
                total_samples.extend(self.build_samples_stream(scene, curr_frame_count))
            elif self.sample_mode=="cluster":
                total_samples.extend(self.build_samples_cluster(scene))
            frame_count += curr_frame_count

            # load pose and intrinsics for current scene
            self.load_intrinsics(scene)

        return (total_samples, frame_count)


    def build_samples_stream(self, scene: str, frame_count: int) -> list[dict[str, Any]]:
        samples: list[dict[str,Any]] = []

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

            image_files: list[str] = []
            depth_files: list[str] = []
            pose_files: list[str] = []
            for ind in frame_inds:
                image_files.append(os.path.join(self.data_path, "Images", scene, f"{ind:08d}.png"))
                depth_files.append(os.path.join(self.data_path, "Depths", scene, f"{ind:08d}_depth.pfm"))
                pose_files.append(os.path.join(self.data_path, "Cameras", f"{ind:08d}_cam.txt"))
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

    def get_cluster_file(self, scene: str) -> str:
        return os.path.join(self.data_path, "Cameras/pair.txt")

    def build_samples_cluster(self, scene: str) -> list[dict[str, Any]]:
        samples: list[dict[str,Any]] = []
        clusters = read_cluster_list(self.get_cluster_file(scene))

        for c in clusters:
            skip = False
            frame_inds = [c[0]]
            if self.random_clusters:
                src_frames = np.random.permutation(c[1])
            else:
                src_frames = c[1]
            frame_inds.extend(src_frames[:self.num_frame-1])

            image_files: list[str] = []
            depth_files: list[str] = []
            pose_files: list[str] = []
            for ind in frame_inds:
                image_files.append(os.path.join(self.data_path, "Images", scene, f"{ind:08d}.png"))
                depth_files.append(os.path.join(self.data_path, "Depths", scene, f"{ind:08d}_depth.pfm"))
                pose_files.append(os.path.join(self.data_path, "Cameras", f"{ind:08d}_cam.txt"))
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


    def load_intrinsics(self, scene: str) -> None:
        cam_file = os.path.join(self.data_path, "Cameras/00000000_cam.txt")
        cam = read_single_cam_sfm(cam_file)
        self.K[scene] = cam[1,:3,:3].astype(np.float32)
        self.K[scene][0,2] -= (self.crop_w//2)
        self.K[scene][1,2] -= (self.crop_h//2)
        self.H: int = int(self.scale * (self.cfg["camera"]["height"] - self.crop_h))
        self.W: int = int(self.scale * (self.cfg["camera"]["width"]- self.crop_w))

    def get_pose(self, pose_file: str) -> NDArray[np.float32]:
        try:
            cam = read_single_cam_sfm(pose_file)
        except:
            print(f"Pose file {pose_file} does not exist.")
            sys.exit()
        pose = cam[0]
        if(np.isnan(pose).any()):
            print(pose, pose_file)
        return pose

    def get_all_poses(self) -> list[NDArray[Any]]:
        poses: list[NDArray[Any]] = []
        pose_path = os.path.join(self.data_path, "Cameras")
        pose_files = os.listdir(pose_path)
        pose_files.sort()
        for pose_file in pose_files:
            if (pose_file[-8:] != "_cam.txt"):
                continue
            cam = read_single_cam_sfm(os.path.join(pose_path,pose_file))
            pose = cam[0]
            if(np.isnan(pose).any()):
                print(pose, pose_file)
            poses.append(pose)
        return poses

    def get_all_depths(self, scale: bool) -> NDArray[Any]:
        gt_depth_files = os.listdir(self.gt_depth_path)
        gt_depth_files = [os.path.join(self.gt_depth_path, gdf) for gdf in gt_depth_files if os.path.isfile(os.path.join(self.gt_depth_path, gdf))]
        gt_depth_files.sort()

        gt_depths = {}
        for gdf in gt_depth_files:
            ref_ind = int(gdf[-18:-10])
            gt_depth = self.get_depth(os.path.join(self.gt_depth_path, gdf), scale=scale)
            gt_depths[ref_ind] = gt_depth
        return gt_depths
