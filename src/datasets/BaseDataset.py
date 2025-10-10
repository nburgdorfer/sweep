import numpy as np
from torch import Tag
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
        self.set_data_paths()
        self.device = self.cfg["device"]
        self.scenes = scenes
        self.divisible_factor = self.cfg["camera"]["divisible_factor"]
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
            self.random_crop = self.cfg["training"]["random_crop"]["enable"]
            self.random_crop_height = self.cfg["training"]["random_crop"]["height"]
            self.random_crop_width = self.cfg["training"]["random_crop"]["width"]
        self.random_clusters = mode == "training"

        if self.random_crop:
            self.H = self.random_crop_height
            self.W = self.random_crop_width
        else:
            self.H = int(
                (self.cfg["camera"]["height"] * self.scale // self.divisible_factor)
                * self.divisible_factor
            )
            self.W = int(
                (self.cfg["camera"]["width"] * self.scale // self.divisible_factor)
                * self.divisible_factor
            )

        self.samples, self.frame_count = self.build_samples()

    def set_data_paths(self, *args, **kwargs):
        raise NotImplementedError()

    def build_samples(self, *args, **kwargs):
        raise NotImplementedError()

    def get_camera_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def custom_crop(self, image, start_row, start_col, crop_height, crop_width):
        return image[
            start_row : start_row + crop_height, start_col : start_col + crop_width
        ]

    def center_crop(
        self,
        image: NDArray[np.float32],
        new_shape: tuple[int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
    ) -> NDArray[np.float32]:
        """Crops the center of an image.

        Either new_shape or crop_size should be provided to this function.
        The new_shape argument specifies the new shape of the image to be
        returned. This should be a tuple of (height, width). The crop_size
        argument specifies the amount of rows and columns to be cropped from
        the input image. This should be a tuple of (crop_rows, crop_cols).
        If both are provided, this function defaults to cropping based on the
        new_shape argument.

        Args:
            image: The image to be cropped.
            new_shape: The desired size for the cropped image (height, width)
            crop_size: The desired amount to be cropped (crop_rows, crop_cols)
        """
        if new_shape is not None:
            crop_height = image.shape[0] - new_shape[0]
            crop_width = image.shape[1] - new_shape[1]
        elif crop_size is not None:
            crop_height = crop_size[0]
            crop_width = crop_size[1]
        else:
            raise Exception(
                "ERROR: Must provide a valid image or cropping shape for cetner cropping."
            )

        return image[
            (crop_height // 2) : image.shape[0] - (crop_height // 2),
            (crop_width // 2) : image.shape[1] - (crop_width // 2),
        ]

    def factor_crop(self, image, factor):
        h, w = image.shape[0], image.shape[1]

        new_h = int((h // factor) * factor)
        new_w = int((w // factor) * factor)

        crop_row = h - new_h
        crop_col = w - new_w

        return (
            self.center_crop(image, crop_size=(crop_row, crop_col)),
            crop_row,
            crop_col,
        )

    def normalize(self, image):
        return ((image / 255.0) - 0.5) * 2.0

    def corner_crop_image(self, image, crop_height, crop_width):
        return image[:crop_height, crop_width]

    def scale_image(self, image, scale):
        return cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_valid_random_crop(self, depth, crop_height, crop_width, iterations=1000):
        for _ in range(iterations):
            crop_row = np.random.randint(0, depth.shape[0] - crop_height)
            crop_col = np.random.randint(0, depth.shape[1] - crop_width)
            cropped_depth = self.custom_crop(
                depth, crop_row, crop_col, crop_height, crop_width
            )

            if np.any(cropped_depth > 0.0):
                return crop_row, crop_col

        raise Exception(
            "ERROR: could not find a valid random crop that includes any ground truth data."
        )

    def get_image(
        self, image_file: str, convert_to_rgb: bool = False
    ) -> NDArray[np.float32]:
        image = cv2.imread(image_file)

        if convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def get_depth(self, depth_file: str) -> NDArray[np.float32]:
        if depth_file[-3:] == "pfm":
            depth = read_pfm(depth_file)
        elif self.png_depth_scale is not None:
            depth = cv2.imread(depth_file, 2) / self.png_depth_scale
        else:
            raise Exception(
                f"ERROR: If depth file is not a 'pfm', an rgb->depth scale must be provided."
            )

        return depth.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        idx = idx % self.__len__()
        sample = self.samples[idx]

        images = [None] * self.num_frame
        target_depths = [None] * self.num_frame
        extrinsics = [None] * self.num_frame
        intrinsics = [None] * self.num_frame
        target_depth = None
        crop_row = 0
        crop_col = 0
        for i in range(len(sample["frame_inds"])):
            image = self.get_image(sample["image_files"][i])
            target_depth = self.get_depth(sample["depth_files"][i])
            P_i, K_i = self.get_camera_parameters(sample["camera_files"][i])

            if self.random_crop:
                if i == 0:
                    # random crop images
                    crop_row, crop_col = self.get_valid_random_crop(
                        target_depth, self.random_crop_height, self.random_crop_width
                    )

                target_depth = self.custom_crop(
                    target_depth,
                    crop_row,
                    crop_col,
                    self.random_crop_height,
                    self.random_crop_width,
                )

                K_i = crop_intrinsics(K_i, crop_row, crop_col)
                image = self.custom_crop(
                    image,
                    crop_row,
                    crop_col,
                    self.random_crop_height,
                    self.random_crop_width,
                )

            elif self.scale < 1.0:
                K = scale_intrinsics(K_i, scale=self.scale)
                image = self.scale_image(image, self.scale)
                target_depth = self.scale_image(target_depth, self.scale)

            # crop according to the desired factor of divisibility
            # (used to make resolution a multiple of self.devisible_factor)
            image, crop_h, crop_w = self.factor_crop(image, self.divisible_factor)
            K_i = crop_intrinsics(K_i, crop_h // 2, crop_w // 2)
            image = self.normalize(image)
            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
            
            target_depth = self.center_crop(
                target_depth, crop_size=(crop_h, crop_w)
            )
            target_depth = target_depth.reshape(
                1, target_depth.shape[0], target_depth.shape[1]
            )

            images[i] = image
            target_depths[i] = target_depth
            extrinsics[i] = P_i
            intrinsics[i] = K_i
        images = np.asarray(images, dtype=np.float32)
        target_depths = np.asarray(target_depths, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)

        # compute min and max camera baselines
        min_baseline, max_baseline = compute_baselines(extrinsics)

        # load data dict
        data = {}
        data["ref_id"] = int(sample["frame_inds"][0])
        data["images"] = images
        data["extrinsics"] = extrinsics
        # data["intrinsics"] = intrinsics
        data["K"] = intrinsics[0]
        data["target_depth"] = target_depths[0]
        data["all_target_depths"] = target_depths
        if self.cfg["camera"]["baseline_mode"] == "min":
            data["baseline"] = min_baseline
        elif self.cfg["camera"]["baseline_mode"] == "max":
            data["baseline"] = max_baseline

        ## Scaling intrinsics for the feature pyramid
        if self.resolution_levels > 1:
            assert data["target_depth"] is not None
            data["target_depths"] = build_depth_pyramid(
                data["target_depth"][0], levels=self.resolution_levels
            )

            multires_intrinsics = []
            for i in range(self.num_frame):
                multires_intrinsics.append(
                    intrinsic_pyramid(intrinsics[0], self.resolution_levels)
                )

            data["multires_intrinsics"] = np.stack(multires_intrinsics).astype(
                np.float32
            )

        return data
