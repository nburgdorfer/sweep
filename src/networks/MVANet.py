""""""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtkit.geometry import uniform_hypothesis, homography_warp

from src.components.encoders import FPN
from src.components.regularizers import CostRegNet, PixelwiseNet


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.resolution_stages = self.cfg["model"]["resolution_stages"]
        self.max_stages = len(self.resolution_stages)
        self.mode = self.cfg["mode"]
        self.group_channels = self.cfg["model"]["group_channels"]
        self.feature_channels = self.cfg["model"]["feature_channels"]
        self.resolution_levels = len(self.feature_channels)
        self.depth_near = self.cfg["camera"]["near"]
        self.depth_far = self.cfg["camera"]["far"]
        self.height, self.width = self.cfg["H"], self.cfg["W"]

        #### Image Feature Encoder
        self.feature_encoder = FPN(
            in_channels=3,
            out_channels=self.feature_channels,
            base_channels=8,
            levels=4,
            block_size=3,
        )

        #### Cost Volume Regularizers
        self.cost_reg = nn.ModuleList(
            [
                CostRegNet(self.group_channels[i], 8)
                for i in range(self.resolution_levels)
            ]
        )

        #### View Weight Network
        self.view_weight_nets = nn.ModuleList(
            [
                PixelwiseNet(self.group_channels[i])
                for i in range(self.resolution_levels)
            ]
        )

    def build_features(self, data):
        num_views = data["images"].shape[1]
        images = data["images"]

        multi_res_features = self.feature_encoder(images[:, 0]) # coarse -> fine (highest res is last).
        for r, features in enumerate(multi_res_features):
                multi_res_features[r] = multi_res_features[r].unsqueeze(1)

        for v in range(1, num_views):
            features_v = self.feature_encoder(images[:, v]) # fine -> coarse (lowest res is last).

            # concatenate image features for view v at each resolution level
            for r, features in enumerate(features_v):
                multi_res_features[r] = torch.cat((multi_res_features[r], features.unsqueeze(1)), dim=1)

        return multi_res_features

    def subdivide_hypotheses(self, hypotheses, pred_hypo_index, iteration):
        with torch.no_grad():
            batch_size, planes, height, width = hypotheses.shape
            selected_depth = torch.gather(
                hypotheses, dim=1, index=pred_hypo_index.unsqueeze(1)
            ).squeeze(1)

            bin_spacing = torch.abs(hypotheses[:, 1] - hypotheses[:, 0]) / planes
            offset = torch.tensor([-3, -1, 1, 3]).to(hypotheses)
            offset = offset.reshape(1, 4, 1, 1).repeat(
                batch_size, 1, height, width
            ) * bin_spacing.unsqueeze(1)
            next_hypotheses = selected_depth.unsqueeze(1).repeat(1, 4, 1, 1) + offset

            # shift search space up if out of lower bound
            shift_up = torch.where(
                next_hypotheses < self.depth_near,
                (self.depth_near - next_hypotheses),
                0.0,
            )
            next_hypotheses = next_hypotheses + shift_up.max(dim=1, keepdim=True)[0]

            # shift search space down if out of upper bound
            shift_down = torch.where(
                next_hypotheses > self.depth_far,
                (next_hypotheses - self.depth_far),
                0.0,
            )
            next_hypotheses = next_hypotheses - shift_down.max(dim=1, keepdim=True)[0]

            # increase resolution if next stage increases resolution
            if iteration + 1 < self.max_stages:
                if (
                    self.resolution_stages[iteration + 1]
                    < self.resolution_stages[iteration]
                ):
                    next_hypotheses = F.interpolate(
                        next_hypotheses, scale_factor=2, mode="nearest"
                    )

        return next_hypotheses

    def forward(self, data, resolution_stage, iteration, reference_index):
        batch_size, _, _, height, width = data["images"].shape
        output = {}

        # build image features
        if data["image_features"] is None:
            data["image_features"] = self.build_features(data)

        batch_size, _, _, height, width = data["image_features"][resolution_stage].shape

        if iteration == 0:
            hypotheses, _, _ = uniform_hypothesis(
                self.cfg,
                self.device,
                batch_size,
                self.depth_near,
                self.depth_far,
                height,
                width,
                4,
                inv_depth=False,
                bin_format=True,
            )

        else:
            hypotheses = data["hypotheses"][iteration].unsqueeze(1)

        #### Build cost volume ####
        cost_volume = homography_warp(
            features=data["image_features"][resolution_stage],
            intrinsics=data["multires_intrinsics"][:, :, resolution_stage],
            extrinsics=data["extrinsics"],
            hypotheses=hypotheses,
            group_channels=self.group_channels[resolution_stage],
            vwa_net=self.view_weight_nets[resolution_stage],
            reference_index=reference_index,
        )

        #### Cost Regularization ####
        cost_volume = self.cost_reg[resolution_stage](cost_volume)
        cost_volume = cost_volume.squeeze(1)

        # gather depth and subdivide depth hypotheses
        pred_hypo_index = torch.argmax(cost_volume, dim=1).to(torch.int64)
        hypotheses = hypotheses.squeeze(1)
        next_hypotheses = self.subdivide_hypotheses(
            hypotheses, pred_hypo_index, iteration
        )

        # upsample depth and confidence maps to full resolution
        depth = (torch.softmax(cost_volume, dim=1)*hypotheses).sum(dim=1, keepdim=True)
        confidence = torch.max(
            torch.softmax(cost_volume, dim=1), dim=1, keepdim=True
        )[0]
        if (height, width) != (self.height, self.width):
            depth = F.interpolate(
                depth, size=(self.height, self.width), mode="bilinear"
            )
            confidence = F.interpolate(
                confidence, size=(self.height, self.width), mode="bilinear"
            )

        output["hypotheses"] = hypotheses
        output["next_hypotheses"] = next_hypotheses
        output["cost_volume"] = cost_volume
        output["pred_hypo_index"] = pred_hypo_index
        output["final_depth"] = depth
        output["confidence"] = confidence

        return output
