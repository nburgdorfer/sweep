import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from cvtkit.geometry import uniform_hypothesis, homography_warp_var
from cvtkit.camera import scale_intrinsics

from src.components.encoders import BasicEncoder
from src.components.refiners import BasicRefiner
from src.components.regularizers import BasicRegularizer


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.mode = self.cfg["mode"]
        self.depth_near = self.cfg["camera"]["near"]
        self.depth_far = self.cfg["camera"]["far"]
        self.feature_channels = self.cfg["model"]["feature_channels"]

        if self.mode == "inference":
            self.depth_planes = self.cfg["inference"]["depth_planes"]
        else:
            self.depth_planes = self.cfg["training"]["depth_planes"]

        # Image Feature Encoder
        self.feature_extractor = BasicEncoder(
            in_channels=3, c=8, out_channels=self.feature_channels
        )

        # Cost Volume Regularizer
        self.regularizer = BasicRegularizer(in_channels=32, c=8)

    def forward(self, data, reference_index=0):
        images = data["images"]
        K = scale_intrinsics(data["K"], scale=3)
        assert isinstance(K, torch.Tensor)
        intrinsics = (
            K.unsqueeze(1).repeat(1, images.shape[1], 1, 1)
        )
        extrinsics = data["extrinsics"]

        # Extract features
        ref_feature = self.feature_extractor(images[:, 0])
        image_features = ref_feature.unsqueeze(1)
        for idx in range(1, images.shape[1]):
            image_features = torch.cat((image_features,self.feature_extractor(images[:, idx]).unsqueeze(1)), dim=1)

        # Develop hypotheses
        batch_size, _, _, height, width = image_features.shape
        hypotheses, _, _ = uniform_hypothesis(
            self.cfg,
            self.device,
            batch_size,
            self.depth_near,
            self.depth_far,
            height,
            width,
            self.depth_planes,
            inv_depth=True,
        )

        # Build cost volume
        cost_volume = homography_warp_var(
            image_features,
            intrinsics,
            extrinsics,
            hypotheses,
            reference_index,
        )

        # Cost Regularization
        cost_volume = self.regularizer(cost_volume)
        cost_volume = F.softmax(cost_volume, dim=2)

        # Calculate confidence
        confidence = F.max_pool3d(cost_volume.squeeze(1), kernel_size=(cost_volume.shape[2],1,1))

        # Depth regression
        _, _, d, h, w = cost_volume.shape
        hypotheses = F.interpolate(hypotheses, size=(d, h, w))
        regressed_depth = torch.sum(cost_volume * hypotheses, dim=2)

        ## Return
        outputs = {}
        outputs["confidence"] = confidence
        outputs["final_depth"] = regressed_depth

        return outputs
