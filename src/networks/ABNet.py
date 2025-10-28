import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from cvtkit.geometry import uniform_hypothesis, homography_warp
from src.utils.geometry import disparity_hypothesis_planes

from src.components.encoders import FPN
# from src.components.refiners import BasicRefiner
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
        self.z_near = self.cfg["camera"]["near"]
        self.z_far = self.cfg["camera"]["far"]
        self.height, self.width = self.cfg["H"], self.cfg["W"]
        self.z_planes = self.cfg["model"]["z_planes"]
        self.disparity = self.cfg["model"]["disparity"]

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

        # #### Depth Refiner
        # self.refiner = BasicRefiner(in_channels=4, c=8)

    def build_features(self, data, resolution_level):
        views = data["images"].shape[1]
        images = data["images"]

        image_features = None
        for i in range(views):
            image_i = images[:, i]
            if image_features is None:
                image_features = self.feature_encoder(image_i, resolution_level).unsqueeze(1)
            else:
                image_features = torch.cat((image_features, self.feature_encoder(image_i, resolution_level).unsqueeze(1)), dim=1)
        assert image_features is not None
        return image_features

    def get_selected_plane(self, hypotheses, pred_hypo_index, iteration):
        with torch.no_grad():
            selected_plane = torch.gather(
                hypotheses, dim=1, index=pred_hypo_index.unsqueeze(1)
            )

            # increase resolution if next stage increases resolution
            if iteration + 1 < self.max_stages:
                if (
                    self.resolution_stages[iteration + 1]
                    < self.resolution_stages[iteration]
                ):
                    selected_plane = F.interpolate(
                        selected_plane, scale_factor=2, mode="nearest"
                    )
                    
        return selected_plane.squeeze(1)

    def forward(self, data, resolution_stage, iteration, final_iteration, reference_index=0):
        batch_size, _, _, height, width = data["images"].shape
        output = {}

        # build image features
        image_features = self.build_features(data, resolution_stage)
        batch_size, _, height, width = image_features[:,reference_index].shape

        # build hypothesis volume
        if iteration == 0:
            z = torch.tensor([(self.z_far+self.z_near)/2]).reshape(1,1,1).repeat(batch_size, height, width).to(self.device)
        else:
            z = data["selected_plane"][iteration]
        f = data["multires_intrinsics"][:, 0, resolution_stage, 0, 0]
        hypotheses = disparity_hypothesis_planes(z, data["baseline"], f, self.disparity[iteration], self.z_planes[iteration])

        #### Build cost volume ####
        cost_volume = homography_warp(
            features=image_features,
            intrinsics=data["multires_intrinsics"][:, :, resolution_stage],
            extrinsics=data["extrinsics"],
            hypotheses=hypotheses,
            group_channels=self.group_channels[resolution_stage],
            vwa_net=self.view_weight_nets[resolution_stage],
            reference_index=reference_index,
        )

        #### Max Pool conditioned on feature vector norm ###
        channels = cost_volume.shape[1]
        output_bins = 4
        kernel_size = int(self.z_planes[iteration] // output_bins)
        _, indices = F.max_pool3d(torch.norm(torch.clone(cost_volume), dim=1, keepdim=True), kernel_size=(kernel_size,1,1), stride=(kernel_size,1,1), padding=(0,0,0), return_indices=True)
        cost_volume = (torch.flatten(cost_volume,2)[:,:,torch.flatten(indices,2)]).reshape(batch_size, channels, output_bins, height, width)
        max_hypotheses = (torch.flatten(hypotheses,1)[:,torch.flatten(indices,2)]).reshape(batch_size, output_bins, height, width)

        #### Cost Regularization ####
        cost_volume = self.cost_reg[resolution_stage](cost_volume)
        cost_volume = cost_volume.squeeze(1)
        
        # gather depth and subdivide depth hypotheses
        pred_hypo_index = torch.argmax(cost_volume, dim=1).to(torch.int64)
        hypotheses = hypotheses.squeeze(1)
        selected_plane = self.get_selected_plane(
            hypotheses, pred_hypo_index, iteration
        )

        # upsample depth and confidence maps to full resolution
        depth = (torch.softmax(cost_volume, dim=1)*max_hypotheses).sum(dim=1, keepdim=True)
        confidence = torch.max(
            torch.softmax(cost_volume, dim=1), dim=1, keepdim=True
        )[0]
        if (height, width) != (self.height, self.width):
            # depth = F.interpolate(
            #     depth, size=(self.height, self.width), mode="nearest"
            # )
            confidence = F.interpolate(
                confidence, size=(self.height, self.width), mode="nearest"
            )

        # # Depth Refinement
        # if final_iteration:
        #     ref_image =  data["images"][:,0]
        #     depth = self.refiner(ref_image, depth)

        output["max_hypotheses"] = max_hypotheses
        output["selected_plane"] = selected_plane
        output["cost_volume"] = cost_volume
        output["pred_hypo_index"] = pred_hypo_index
        output["final_depth"] = depth
        output["confidence"] = confidence

        return output
