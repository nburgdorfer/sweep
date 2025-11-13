import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtkit.geometry import uniform_hypothesis, homography_warp
from cvtkit.common import top_k_hypothesis_selection, laplacian_pyramid

from src.components.encoders import FPN
from src.components.regularizers import DenseCostReg, SparseCostReg, ViewWeightAgg #, PixelwiseNet


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
            block_size=3,
            levels=4,
        )

        #### Cost Volume Regularizers ####
        regularizer = []
        regularizer.extend(
            [
                SparseCostReg(self.group_channels[i])
                for i in range(0, self.resolution_levels-1)
            ]
        )
        regularizer.extend([DenseCostReg(self.group_channels[self.resolution_levels-1])])
        self.regularizer = nn.ModuleList(regularizer)

        #### View Weight Aggregation Network ####
        # self.view_weight_nets = ViewWeightAgg(self.group_channels[-1])
        # self.view_weight_nets = nn.ModuleList(
        #     [
        #         PixelwiseNet(self.group_channels[i])
        #         for i in range(self.resolution_levels)
        #     ]
        # )
        self.view_weight_nets = nn.ModuleList(
            [
                ViewWeightAgg(self.group_channels[i])
                for i in range(self.resolution_levels)
            ]
        )

    def build_features(self, data):
        batch_size, views, _, height, width = data["images"].shape
        images = data["images"]

        image_features_lists = {i: [] for i in range(self.resolution_levels)}
        for i in range(views):
            image_i = images[:, i]
            features_i = self.feature_encoder(image_i)
            for j in range(self.resolution_levels):
                image_features_lists[j].append(features_i[j])

        image_features = []
        for j in range(self.resolution_levels):
            image_features.append(torch.stack(image_features_lists[j], dim=1))            

        return image_features

    def forward(self, data):
        batch_size, _, _, height, width = data["images"].shape

        #### Encode Images ####
        image_features = self.build_features(data,)

        hypos = [None] * self.resolution_levels
        hypo_coords = [None] * self.resolution_levels
        intervals = [None] * self.resolution_levels
        prob_grids = [None] * self.resolution_levels
        for level in range(self.resolution_levels-1, -1, -1):
            batch_size, _, _, height, width = image_features[level].shape
            if level == self.resolution_levels-1:
                hypos[level], hypo_coords[level], intervals[level] = uniform_hypothesis(
                    self.cfg,
                    self.device,
                    batch_size,
                    self.depth_near,
                    self.depth_far,
                    height,
                    width,
                    self.depth_planes[level],
                    inv_depth=False,
                    bin_format=False,
                )
            else:
                hypos[level], hypo_coords[level], intervals[level] = (
                    top_k_hypothesis_selection(
                        prob_grids[level + 1],
                        k=int(self.depth_planes[level] / 2),
                        prev_hypos=hypos[level + 1],
                        prev_hypo_coords=hypo_coords[level + 1],
                        prev_intervals=intervals[level + 1],
                    )
                )

            #### Build cost volume ####
            cost_volume = homography_warp(
                features=image_features[level],
                intrinsics=data["multires_intrinsics"][:, :, level],
                extrinsics=data["extrinsics"],
                hypotheses=hypos[level],
                group_channels=self.group_channels[level],
                vwa_net = self.view_weight_nets[level],
                reference_index=0,
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
        # maximum_prob, max_prob_idx = prob_grids[0].max(dim=2)
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
