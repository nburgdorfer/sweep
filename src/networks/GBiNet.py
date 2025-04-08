import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtkit.geometry import uniform_hypothesis, homography_warp
from cvtkit.common import top_k_hypothesis_selection, laplacian_pyramid
from cvtkit.camera import intrinsic_pyramid
from cvtkit.io import load_ckpt

from src.components.encoders import FPN_small
from src.components.regularizers import CostRegNet, PixelwiseNet

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.stage_ids = self.cfg["model"]["stage_ids"]
        self.iterations = len(self.stage_ids)
        self.mode = self.cfg["mode"]
        self.group_channels = self.cfg["model"]["group_channels"]
        self.feature_channels = self.cfg["model"]["feature_channels"]
        self.resolution_levels = len(self.feature_channels)
        self.depth_near = self.cfg["camera"]["near"]
        self.depth_far = self.cfg["camera"]["far"]
        self.height, self.width = self.cfg["H"], self.cfg["W"]

        #### Image Feature Encoder
        self.feature_encoder = FPN_small(3, 8, self.feature_channels)

        #### Cost Volume Regularizers
        self.cost_reg = nn.ModuleList([CostRegNet(self.group_channels[i], 8) for i in range(self.resolution_levels)])
        
        #### View Weight Network
        self.view_weight_nets = nn.ModuleList([PixelwiseNet(self.group_channels[i]) for i in range(self.resolution_levels)])

    def build_features(self, data):
        batch_size, views, _, height, width = data["images"].shape
        images = data["images"]

        image_features = {i:[] for i in range(self.resolution_levels)}
        for i in range(views):
            image_i = images[:, i]
            features_i = self.feature_encoder(image_i)
            for j in range(self.resolution_levels):
                image_features[j].append(features_i[j])

        return image_features

    def subdivide_hypotheses(self, hypotheses, pred_hypo_index, iteration):
        with torch.no_grad():
            batch_size, planes, height, width = hypotheses.shape
            selected_depth = torch.gather(hypotheses, dim=1, index=pred_hypo_index.unsqueeze(1)).squeeze(1)

            bin_spacing = torch.abs(hypotheses[:,1] - hypotheses[:,0]) / planes
            offset = torch.tensor([-3,-1,1,3]).to(hypotheses)
            offset = offset.reshape(1,4,1,1).repeat(batch_size,1,height,width) * bin_spacing.unsqueeze(1)
            next_hypotheses = selected_depth.unsqueeze(1).repeat(1,4,1,1) + offset

            # shift search space up if out of lower bound
            shift_up = torch.where(next_hypotheses < self.depth_near, (self.depth_near-next_hypotheses), 0.0)
            next_hypotheses = next_hypotheses + shift_up.max(dim=1, keepdim=True)[0]

            # shift search space down if out of upper bound
            shift_down = torch.where(next_hypotheses > self.depth_far, (next_hypotheses - self.depth_far), 0.0)
            next_hypotheses = next_hypotheses - shift_down.max(dim=1, keepdim=True)[0]

            # increase resolution if next stage increases resolution
            if iteration+1 < self.iterations:
                if (self.stage_ids[iteration+1] > self.stage_ids[iteration]):
                    next_hypotheses = F.interpolate(next_hypotheses, scale_factor=2, mode="nearest")
        
        return next_hypotheses

    def forward(self, data, stage_id, iteration, prob_out_depth=6):
        batch_size, views, _, height, width = data["images"].shape
        output = {}

        # build image features
        image_features = self.build_features(data)[stage_id]
        batch_size, channels, height, width = image_features[0].shape

        if iteration==0:
            hypotheses, _, _ = uniform_hypothesis(
                self.cfg,
                self.device,
                batch_size,
                self.depth_near,
                self.depth_far,
                height, width,
                4,
                inv_depth=False,
                bin_format=True
            )

        else:
            hypotheses = data["hypotheses"][iteration]

        #### Build cost volume ####
        cost_volume, _ = homography_warp(
            self.cfg,
            image_features,
            data["multires_intrinsics"][:,:,stage_id], 
            data["poses"],
            hypotheses,
            self.group_channels[stage_id],
            self.view_weight_nets[stage_id],
            None,
            False
        )

        #### Cost Regularization ####
        cost_volume = self.cost_reg[stage_id](cost_volume)
        cost_volume = cost_volume.squeeze(1)

        # gather depth and subdivide depth hypotheses
        pred_hypo_index = torch.argmax(cost_volume, dim=1).to(torch.int64)
        hypotheses = hypotheses.squeeze(1)
        depth = torch.gather(hypotheses, dim=1, index=pred_hypo_index.unsqueeze(1))
        confidence = torch.max(torch.softmax(cost_volume, dim=1), dim=1, keepdim=True)[0]
        next_hypotheses = self.subdivide_hypotheses(hypotheses, pred_hypo_index, iteration)

        # upsample depth and confidence maps to full resolution
        if (height, width) != (self.height, self.width):
            depth = F.interpolate(depth, size=(self.height, self.width), mode="bilinear")
            confidence = F.interpolate(confidence, size=(self.height, self.width), mode="bilinear")

        output["hypotheses"] = hypotheses
        output["next_hypotheses"] = next_hypotheses
        output["cost_volume"] = cost_volume
        output["pred_hypo_index"] = pred_hypo_index
        output["final_depth"] = depth
        output["confidence"] = confidence

        return output
