import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtkit.geometry import uniform_hypothesis, homography_warp_var
from cvtkit.common import top_k_hypothesis_selection
from cvtkit.camera import scale_intrinsic
from cvtkit.io import load_ckpt

from src.components.encoders import BasicEncoder
from src.components.refiners import BasicRefiner
from src.components.regularizers import BasicRegularizer

class Network(nn.Module):
    def __init__(self, cfg, device):
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = device
        self.mode = self.cfg["mode"]
        self.depth_near = self.cfg["camera"]["near"]
        self.depth_far = self.cfg["camera"]["far"]
        self.feature_channels = self.cfg["model"]["feature_channels"]

        if self.mode=="inference":
            self.depth_planes = self.cfg["inference"]["depth_planes"]
        else:
            self.depth_planes = self.cfg["training"]["depth_planes"]

        # Image Feature Encoder
        self.feature_extractor = BasicEncoder(in_channels=3, c=8, out_channels=self.feature_channels)

        # Cost Volume Regularizer
        self.regularizer = BasicRegularizer(in_channels=32, c=8)

        # Depth Refiner
        self.refiner = BasicRefiner(in_channels=4, c=32)

    def forward(self, data):
        images =  data["images"]
        intrinsics = scale_intrinsic(data["K"], s=3).unsqueeze(1).repeat(1,images.shape[1],1,1)
        extrinsics = data["poses"]

        # Extract features
        ref_feature = self.feature_extractor(images[:,0])
        image_features = [ref_feature]
        for idx in range(1, images.shape[1]):
            src_feature = self.feature_extractor(images[:,idx])
            image_features.append(src_feature)

        # Develop hypotheses
        batch_size, channels, height, width = image_features[0].shape
        hypotheses, _, _ = uniform_hypothesis(
            self.cfg,
            self.device,
            batch_size,
            self.depth_near,
            self.depth_far,
            height, width,
            self.depth_planes,
            inv_depth=True
        )

        # Build cost volume
        cost_volume = homography_warp_var(
            self.cfg,
            image_features,
            intrinsics[:,0], 
            intrinsics[:,1:],
            extrinsics[:,0],
            extrinsics[:,1:],
            hypotheses
        )

        # Cost Regularization
        cost_volume = self.regularizer(cost_volume)
        cost_volume = F.softmax(cost_volume,dim=2)

        # Calculate confidence
        maximum_prob, max_prob_idx = cost_volume.max(dim=2)
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(cost_volume, pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
        confidence, _ = prob_volume_sum4.max(dim=1, keepdim=True)

        # Depth regression
        _,_,d,h,w = cost_volume.shape
        hypotheses = F.interpolate(hypotheses, size=(d,h,w))
        regressed_depth = torch.sum(cost_volume*hypotheses,dim=2)

        # resize confidence and depth
        ref_image = data["images"][:,0]
        regressed_depth = F.interpolate(regressed_depth, (ref_image.shape[2],ref_image.shape[3]), mode="bilinear")
        confidence = F.interpolate(confidence, (ref_image.shape[2],ref_image.shape[3]), mode="bilinear")

        # Depth Refinement
        refined_depth = self.refiner(ref_image, regressed_depth)

        ## Return
        outputs = {}
        outputs["confidence"] = confidence
        outputs["final_depth"] = refined_depth

        return outputs
