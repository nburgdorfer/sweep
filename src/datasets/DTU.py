import os
import numpy as np
from tqdm import tqdm

from typing import Any, Dict
from numpy.typing import NDArray

from cvtkit.io import read_single_cam_sfm, read_cluster_list

from src.datasets.BaseDataset import BaseDataset


class DTU(BaseDataset):
    def __init__(self, cfg: dict[Any, Any], mode: str, scenes: list[str]):
        super(DTU, self).__init__(cfg, mode, scenes)

    def set_data_paths(self):
        self.images_path = os.path.join(self.data_path, "Image_Lightings")
        self.target_depths_path = os.path.join(self.data_path, "Depths")
        self.cameras_path = os.path.join(self.data_path, "Cameras")

    def get_frame_count(self, scene: str):
        image_files = os.listdir(os.path.join(self.images_path, scene))
        image_files = [img for img in image_files if img[-4:] == ".png"]
        return len(image_files)

    def build_samples(self) -> tuple[list[dict[str, Any]], int]:
        total_samples: list[dict[str, Any]] = []
        frame_count = 0

        if self.mode == "inference":
            scenes = self.scenes
        else:
            scenes = tqdm(self.scenes, desc="Loading scene paths", unit="scene")

        for scene in scenes:
            if not os.path.isdir(os.path.join(self.images_path, scene)):
                print(
                    f"{os.path.join(self.images_path, scene)} is not a valid directory."
                )
                continue

            # build samples dict for other data
            curr_frame_count = self.get_frame_count(scene)
            if self.sample_mode == "stream":
                total_samples.extend(self.build_samples_stream(scene, curr_frame_count))
            elif self.sample_mode == "cluster":
                total_samples.extend(self.build_samples_cluster(scene))
            frame_count += curr_frame_count

        return (total_samples, frame_count)

    def build_samples_stream(
        self, scene: str, frame_count: int
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        frame_offset = (self.num_frame - 1) // 2
        radius = frame_offset * self.frame_spacing
        for ref_frame in range(frame_count):
            skip = False
            if (ref_frame + radius >= frame_count) or (ref_frame - radius < 0):
                continue
            start = ref_frame - radius
            end = ref_frame + radius

            frame_inds = [
                i
                for i in range(ref_frame, end + 1, self.frame_spacing)
                if i != ref_frame
            ]
            bottom = [
                i
                for i in range(ref_frame, start - 1, -self.frame_spacing)
                if i != ref_frame
            ]
            frame_inds.extend(bottom)
            frame_inds.insert(0, ref_frame)

            image_files: list[str] = []
            depth_files: list[str] = []
            camera_files: list[str] = []
            for ind in frame_inds:
                image_files.append(
                    os.path.join(self.images_path, scene, f"{ind:08d}.png")
                )
                depth_files.append(
                    os.path.join(self.target_depths_path, scene, f"{ind:08d}_depth.pfm")
                )
                camera_files.append(
                    os.path.join(self.cameras_path, f"{ind:08d}_cam.txt")
                )
                if not os.path.isfile(camera_files[-1]):
                    # print(f"{camera_files[-1]} does not exist; skipping")
                    skip = True
            if skip:
                continue

            samples.append(
                {
                    "scene": scene,
                    "frame_inds": frame_inds,
                    "image_files": image_files,
                    "depth_files": depth_files,
                    "camera_files": camera_files,
                }
            )
        return samples

    def get_cluster_file(self) -> str:
        return os.path.join(self.cameras_path, "pair.txt")

    def build_samples_cluster(self, scene: str) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        clusters = read_cluster_list(self.get_cluster_file())

        for c in clusters:
            skip = False
            frame_inds = [c[0]]
            if self.random_clusters:
                src_frames = np.random.permutation(c[1])
            else:
                src_frames = c[1]
            frame_inds.extend(src_frames[: self.num_frame - 1])

            image_files: list[str] = []
            depth_files: list[str] = []
            camera_files: list[str] = []
            for ind in frame_inds:
                image_files.append(
                    os.path.join(self.images_path, scene, f"{ind:08d}_3.png")
                )
                depth_files.append(
                    os.path.join(self.target_depths_path, scene, f"{ind:08d}_depth.pfm")
                )
                camera_files.append(
                    os.path.join(self.cameras_path, f"{ind:08d}_cam.txt")
                )
                if not os.path.isfile(camera_files[-1]):
                    # print(f"{camera_files[-1]} does not exist; skipping")
                    skip = True
            if skip:
                continue

            samples.append(
                {
                    "scene": scene,
                    "frame_inds": frame_inds,
                    "image_files": image_files,
                    "depth_files": depth_files,
                    "camera_files": camera_files,
                }
            )
        return samples

    def get_camera_parameters(
        self, camera_file: str
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        try:
            cam = read_single_cam_sfm(camera_file)
        except Exception as e:
            raise Exception(f"Pose file {camera_file} does not exist.\n\n{e}")

        # get extrinsics
        P = cam[0][:4, :4]
        if np.isnan(P).any():
            raise Exception(
                f"ERROR: elements of extrinsic matrix in file '{camera_file}' are NAN. P: {P}"
            )

        # get intrinsics
        K = cam[1][:3, :3]
        if np.isnan(K).any():
            raise Exception(
                f"ERROR: elements of intrinsic matrix in file '{camera_file}' are NAN. K: {K}"
            )
        return (P, K)

    def get_all_camera_parameters(self) -> list[NDArray[Any]]:
        extrinsics: list[NDArray[Any]] = []
        intrinsics: list[NDArray[Any]] = []
        camera_files = os.listdir(self.cameras_path)
        camera_files.sort()
        for camera_file in camera_files:
            if camera_file[-8:] != "_cam.txt":
                continue
            P, K = self.get_camera_parameters(camera_file)
            extrinsics.append(P)
            intrinsics.append(K)
        return extrinsics

    def get_all_depths(self, scene: str) -> Dict[int, NDArray[Any]]:
        target_depths_path = os.path.join(self.target_depths_path, scene)
        target_depth_files = os.listdir(target_depths_path)
        target_depth_files = [
            os.path.join(target_depths_path, tdf)
            for tdf in target_depth_files
            if os.path.isfile(os.path.join(target_depths_path, tdf))
        ]
        target_depth_files.sort()

        target_depths = {}
        for tdf in target_depth_files:
            ref_ind = int(tdf[-18:-10])
            target_depth = self.get_depth(os.path.join(self.target_depths_path, tdf))

            # scale and crop depth
            # crop according to the desired factor of divisibility
            # (used to make resolution a multiple of self.devisible_factor)
            target_depth = self.scale_image(target_depth, self.scale)
            target_depth, _, _ = self.factor_crop(target_depth, self.divisible_factor)

            target_depths[ref_ind] = target_depth
        return target_depths
