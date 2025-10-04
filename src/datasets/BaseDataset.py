import numpy as np
from torch.utils.data import Dataset
import cv2

from typing import Any
from numpy.typing import NDArray

from cvtkit.io import read_pfm
from cvtkit.camera import (
    compute_baselines,
    intrinsic_pyramid,
    scale_intrinsics,
    crop_intrinsics,
)
from cvtkit.common import build_depth_pyramid, normalize, crop_image


def build_dataset(cfg: dict[Any, Any], mode: str, scenes: list[str]):
    if cfg["dataset"] == "TNT":
        from src.datasets.TNT import TNT as Dataset
    elif cfg["dataset"] == "DTU":
        from src.datasets.DTU import DTU as Dataset
    else:
        raise Exception(f"Unknown Dataset {cfg['dataset']}")

    return Dataset(cfg, mode, scenes)


class BaseDataset(Dataset[dict[str, Any]]):
    def __init__(self, cfg: dict[Any, Any], mode: str, scenes: list[str]):
        self.cfg = cfg
        self.mode = mode
        self.data_path = self.cfg["data_path"]
        self.device = self.cfg["device"]
        self.scenes = scenes
        self.crop_h = self.cfg["camera"]["crop_h"]
        self.crop_w = self.cfg["camera"]["crop_w"]
        self.png_depth_scale: float | None = None

        try:
            self.resolution_levels = len(self.cfg["model"]["feature_channels"])
        except:
            self.resolution_levels = 1

        if self.mode == "inference":
            self.num_frame = self.cfg["inference"]["num_frame"]
            self.frame_spacing = self.cfg["inference"]["frame_spacing"]
            self.scale = self.cfg["inference"]["scale"]
            self.sample_mode = self.cfg["inference"]["sample_mode"]
            self.random_crop = False
        else:
            self.num_frame = self.cfg["training"]["num_frame"]
            self.frame_spacing = self.cfg["training"]["frame_spacing"]
            self.scale = self.cfg["training"]["scale"]
            self.sample_mode = self.cfg["training"]["sample_mode"]
            self.random_crop = self.cfg["training"]["random_crop"]
        self.random_clusters = mode == "training"

        self.K: dict[str, Any] = {}
        self.W = 0
        self.H = 0
        self.samples, self.frame_count = self.build_samples()

        # if self.mode == "inference":
        #     # use all samples during inference
        #     self.samples = self.total_samples
        # else:
        #     # shuffle and sub-sample during training
        #     if self.mode=="training":
        #         self.max_samples = self.cfg["training"]["max_training_samples"]
        #     elif self.mode=="validation":
        #         self.max_samples = self.cfg["training"]["max_val_samples"]
        #     self.shuffle_and_subsample()

    def build_samples(self) -> tuple[list[dict[str, Any]], int]:
        raise NotImplementedError()

    def load_intrinsics(self, scene: str) -> None:
        raise NotImplementedError()

    def get_pose(self, pose_file: str) -> NDArray[np.float32]:
        raise NotImplementedError()

    def get_image(self, image_file: str, scale: bool = True):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop and resize image
        h, w, _ = image.shape
        image = image[
            (self.crop_h // 2) : h - (self.crop_h // 2),
            (self.crop_w // 2) : w - (self.crop_w // 2),
            :,
        ]
        if scale:
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        image = normalize(image)
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        return image.astype(np.float32)

    def get_depth(self, depth_file: str, scale: bool = True) -> NDArray[Any]:
        if depth_file[-3:] == "pfm":
            depth = read_pfm(depth_file)
        elif self.png_depth_scale is not None:
            depth = cv2.imread(depth_file, 2) / self.png_depth_scale
        else:
            raise Exception(
                f"ERROR: If depth file is not a 'pfm', an rgb->depth scale must be provided."
            )

        # crop and resize depth
        h, w = depth.shape
        depth = depth[
            (self.crop_h // 2) : h - (self.crop_h // 2),
            (self.crop_w // 2) : w - (self.crop_w // 2),
        ]
        if scale:
            depth = cv2.resize(
                src=depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR
            )
        depth = depth.reshape(1, depth.shape[0], depth.shape[1])
        return depth.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        idx = idx % self.__len__()
        sample = self.samples[idx]
        scene = sample["scene"]

        # load and compute intrinsics
        K = np.copy(self.K[scene])

        # random crop scaled image patches or scale entire image
        if self.random_crop:
            crop_row = np.random.randint(
                0, (self.cfg["camera"]["height"] - self.crop_h) - self.H
            )
            crop_col = np.random.randint(
                0, (self.cfg["camera"]["width"] - self.crop_w) - self.W
            )
            K = crop_intrinsics(K, crop_row, crop_col)
        else:
            K = scale_intrinsics(K, scale=self.scale)

        images = [None] * self.num_frame
        poses = [None] * self.num_frame
        target_depths = [None] * self.num_frame
        filenames = []
        for i, fid in enumerate(sample["frame_inds"]):
            images[i] = self.get_image(
                sample["image_files"][i], scale=(not self.random_crop)
            )
            poses[i] = self.get_pose(sample["pose_files"][i])
            target_depths[i] = self.get_depth(
                sample["depth_files"][i], scale=(not self.random_crop)
            )
            filenames.append(
                scene + "-" + "_".join("%04d" % x for x in sample["frame_inds"])
            )

            if self.random_crop:
                images[i] = crop_image(images[i], crop_row, crop_col, self.scale)
                target_depths[i] = crop_image(
                    target_depths[i], crop_row, crop_col, self.scale
                )
        images = np.asarray(images, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        target_depths = np.asarray(target_depths, dtype=np.float32)

        # compute min and max camera baselines
        min_baseline, max_baseline = compute_baselines(poses)

        # load data dict
        data = {}
        data["ref_id"] = int(sample["frame_inds"][0])
        data["K"] = K
        data["images"] = images
        data["poses"] = poses
        data["target_depth"] = target_depths[0]
        # data["target_depth_1"] = target_depths[1]
        if self.cfg["camera"]["baseline_mode"] == "min":
            data["baseline"] = min_baseline
        elif self.cfg["camera"]["baseline_mode"] == "max":
            data["baseline"] = max_baseline

        ## Scaling intrinsics for the feature pyramid
        if self.resolution_levels > 1:
            data["target_depths"] = build_depth_pyramid(
                data["target_depth"][0], levels=self.resolution_levels
            )

            multires_intrinsics = []
            for i in range(self.num_frame):
                multires_intrinsics.append(
                    intrinsic_pyramid(K, self.resolution_levels)
                )

            data["multires_intrinsics"] = np.stack(multires_intrinsics).astype(
                np.float32
            )

        return data
