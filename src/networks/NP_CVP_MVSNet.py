import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtkit.geometry import uniform_hypothesis, homography_warp
from cvtkit.common import top_k_hypothesis_selection, laplacian_pyramid

from src.components.encoders import FPN
from src.components.regularizers import DenseCostReg, SparseCostReg, ViewWeightAgg


class Network(nn.Module):
    def __init__(self, cfg, device):
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = device
        self.mode = self.cfg["mode"]
        self.group_channels = self.cfg["model"]["group_channels"]
        self.feature_channels = self.cfg["model"]["feature_channels"]
        self.resolution_levels = len(self.feature_channels)
        self.depth_near = self.cfg["camera"]["near"]
        self.depth_far = self.cfg["camera"]["far"]
        if self.mode == "inference":
            self.depth_planes = self.cfg["inference"]["depth_planes"]
        else:
            self.depth_planes = self.cfg["training"]["depth_planes"]

        #### Image Feature Encoder ####
        self.feature_encoder = FPN(
            in_channels=3,
            out_channels=self.feature_channels,
            base_channels=8,
            block_size=5,
            levels=4,
            out_levels=4,
        )

        #### Cost Volume Regularizers ####
        regularizer = [DenseCostReg(self.group_channels[0])]
        regularizer.extend(
            [
                SparseCostReg(self.group_channels[i])
                for i in range(1, self.resolution_levels)
            ]
        )
        self.regularizer = nn.ModuleList(regularizer)

        #### View Weight Aggregation Network ####
        self.vwa_net = ViewWeightAgg(self.group_channels[0])

    def build_features(self, data):
        batch_size, views, _, height, width = data["images"].shape
        images = data["images"]

        image_features = {i: [] for i in range(self.resolution_levels)}
        for i in range(views):
            image_i = images[:, i]
            features_i = self.feature_encoder(image_i)
            for j in range(self.resolution_levels):
                image_features[j].append(features_i[j])

        return image_features

    def forward(self, data):
        batch_size, views, _, height, width = data["images"].shape

        #### Encode Images ####
        image_features = self.build_features(data)

        hypos = [None] * self.resolution_levels
        hypo_coords = [None] * self.resolution_levels
        intervals = [None] * self.resolution_levels
        prob_grids = [None] * self.resolution_levels
        for level in range(self.resolution_levels):

            batch_size, channels, height, width = image_features[level][0].shape
            if level == 0:
                vwa_net = self.vwa_net
                vis_weights = None
                hypos[level], hypo_coords[level], intervals[level] = uniform_hypothesis(
                    self.cfg,
                    self.device,
                    batch_size,
                    self.depth_near,
                    self.depth_far,
                    height,
                    width,
                    self.depth_planes[0],
                    inv_depth=False,
                    bin_format=False,
                )
            else:
                vwa_net = None
                hypos[level], hypo_coords[level], intervals[level] = (
                    top_k_hypothesis_selection(
                        prob_grids[level - 1],
                        k=int(self.depth_planes[level] / 2),
                        prev_hypos=hypos[level - 1],
                        prev_hypo_coords=hypo_coords[level - 1],
                        prev_intervals=intervals[level - 1],
                    )
                )

            #### Build cost volume ####
            cost_volume, vis_weights = homography_warp(
                cfg=self.cfg,
                features=image_features[level],
                intrinsics=data["multires_intrinsics"][:, :, level],
                extrinsics=data["poses"],
                hypotheses=hypos[level],
                group_channels=self.group_channels[level],
                vwa_net=vwa_net,
                vis_weights=vis_weights,
            )

            #### Cost Regularization ####
            occ_grid = self.regularizer[level](
                cost_volume, hypo_coords[level], mode=self.mode
            )

            if self.mode == "inference":
                del cost_volume
                torch.cuda.empty_cache()

            occ_grid = torch.softmax(occ_grid, dim=2)
            prob_grids[level] = occ_grid

        # Calculate confidence
        maximum_prob, max_prob_idx = prob_grids[0].max(dim=2)
        prob_volume_sum4 = 4 * F.avg_pool3d(
            F.pad(prob_grids[0], pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
        ).squeeze(1)
        max_sum4_prob, _ = prob_volume_sum4.max(dim=1)
        confidence = F.interpolate(
            max_sum4_prob.unsqueeze(1),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )

        # Final depth regression
        regressed_depth = torch.sum(prob_grids[-1] * hypos[-1], dim=2)

        torch.cuda.empty_cache()

        # compute laplacians
        image_laplacian = laplacian_pyramid(data["images"][:, 0])
        depth_laplacian = laplacian_pyramid(regressed_depth)

        ## Return
        output = {}
        output["hypos"] = hypos
        output["hypo_coords"] = hypo_coords
        output["intervals"] = intervals
        output["prob_grids"] = prob_grids
        output["confidence"] = confidence
        output["final_depth"] = regressed_depth
        output["image_laplacian"] = image_laplacian
        output["est_depth_laplacian"] = depth_laplacian

        return output
