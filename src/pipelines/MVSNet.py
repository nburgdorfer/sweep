# Python libraries
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

## Custom libraries
from src.pipelines.BasePipeline import BasePipeline
from src.evaluation.eval_2d import depth_acc
from cvt.common import to_gpu, laplacian_pyramid
from cvt.io import write_pfm
from cvt.geometry import visibility, get_uncovered_mask, edge_mask
from cvt.visualization import visualize_mvs, laplacian_depth_error, laplacian_count, laplacian_uncovered_count, plot_laplacian_matrix

# NP-CVP-MVSNet libraries
from src.networks.MVSNet import Network

class Pipeline(BasePipeline):
    def __init__(self, cfg, config_path, log_path, model_name, training_scenes=None, validation_scenes=None, inference_scene=None):
            super(Pipeline, self).__init__(cfg, config_path, log_path, model_name, training_scenes, validation_scenes, inference_scene)

    def get_network(self):
        return Network(self.cfg, self.device).to(self.device)

    def compute_loss(self, data, output):
        loss = {}

        final_depth = output["final_depth"]
        target_depth = F.interpolate(data["target_depth"], (final_depth.shape[2], final_depth.shape[3]))
        #image_laplacian = data["image_laplacian"]
        #depth_laplacian = 4 - data["depth_laplacian"]
        #laplacian_weight = (depth_laplacian+image_laplacian) / 8

        batch_size, c, height, width = target_depth.shape
        assert  final_depth.shape == target_depth.shape, \
                f"Target depth shape ({target_depth.shape}) and Estimate depth shape ({final_depth.shape}) do not match."

        mask = torch.where(target_depth > self.cfg["camera"]["near"], 1.0, 0.0)
        depth_error = F.smooth_l1_loss(final_depth, target_depth, reduction='none') * mask
        #depth_error *= laplacian_weight
        depth_error = depth_error.sum(dim=(1,2,3)) / mask.sum(dim=(1,2,3))

        loss["depth"] = depth_error.mean()
        loss["total"] = loss["depth"]

        return loss

    def compute_stats(self, data, output):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            mae, acc = depth_acc(output["final_depth"][0], data["target_depth"][0])
        stats = {
                "mae": mae,
                "acc": acc
                }
        return stats

    def run(self, mode, epoch):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        if mode == "inference":
            self.mvs_model.eval()
            visualize = self.cfg["inference"]["visualize"]
            vis_freq = self.cfg["inference"]["vis_freq"]
            vis_path = self.vis_path
            data_loader = self.inference_data_loader
            title_suffix = ""
            if visualize:
                M = np.zeros((4,5,5))
        else:
            if mode == "training":
                self.mvs_model.train()
                data_loader = self.training_data_loader
            elif mode == "validation":
                self.mvs_model.eval()
                data_loader = self.validation_data_loader
            visualize = self.cfg["training"]["visualize"]
            vis_freq = self.cfg["training"]["vis_freq"]
            vis_path = self.log_vis
            title_suffix = f" - Epoch {epoch}"
            sums = {
                    "loss": 0.0,
                    "mae": 0.0,
                    "acc": 0.0
                    }


        with tqdm(data_loader, desc=f"MVS-Studio {mode}{title_suffix}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)

                # Run network forward pass
                output = self.mvs_model(data)
                output["image_laplacian"] = laplacian_pyramid(data["images"][:,0], tau=0.1).to(torch.float32)
                output["est_depth_laplacian"] = laplacian_pyramid(output["final_depth"], tau=1.0).to(torch.float32)

                if mode != "inference":
                    # Compute loss
                    loss = self.compute_loss(data, output)

                    # Compute backward pass
                    if mode != "validation":
                        self.mvs_optimizer.zero_grad(set_to_none=True)
                        loss["total"].backward()
                        torch.nn.utils.clip_grad_norm_(self.mvs_model.parameters(), self.cfg["training"]["grad_clip"])
                        self.mvs_optimizer.step()

                    # Compute output statistics
                    stats = self.compute_stats(data, output)

                    # Update progress bar
                    sums["loss"] += float(loss["total"].detach().cpu().item())
                    sums["mae"] += float(stats["mae"].detach().cpu().item())
                    sums["acc"] += float(stats["acc"].detach().cpu().item())
                    max_mem = torch.cuda.max_memory_allocated(device=self.device)
                    max_mem = float(max_mem / 1.073742e9)
                    loader.set_postfix(
                            loss=f"{(sums['loss']/(batch_ind+1)):6.2f}",
                            mae=f"{(sums['mae']/(batch_ind+1)):6.2f}",
                            acc_1cm=f"{(sums['acc']/(batch_ind+1))*100:3.2f}%",
                            max_memory=f"{(max_mem):2.3f}"
                            )

                    ## Log loss and statistics
                    iteration = (len(loader)*(epoch)) + batch_ind
                    self.logger.add_scalar(f"{mode} - Loss", float(loss["total"].detach().cpu().item()), iteration)
                    self.logger.add_scalar(f"{mode} - Mean Average Error", float(stats["mae"].detach().cpu().item()), iteration)
                    self.logger.add_scalar(f"{mode} - Accuracy", float(stats["acc"].detach().cpu().item())*100, iteration)
                    self.logger.add_scalar(f"{mode} - Max Memory", float(max_mem), iteration)
                else:
                    # Store network output
                    self.save_output(data, output, mode, batch_ind, epoch)

                ## Visualization
                if (visualize and batch_ind % vis_freq == 0):
                    data["depth_laplacian"] = laplacian_pyramid(data["target_depth"], tau=1.0)
                    visualize_mvs(data, output, batch_ind, vis_path, self.cfg["visualization"]["max_depth_error"], mode=mode, epoch=epoch)

                if mode=="inference" and visualize:
                    M[0] += laplacian_depth_error(data, output)
                    M[1] += laplacian_depth_error(data, output, use_est_depth=True)
                    M[2] += laplacian_count(data, output)
                    M[3] += laplacian_count(data, output, use_est_depth=True)

                if mode != "inference":
                    del loss
                    del output
                    del data
                    del stats
                    torch.cuda.empty_cache()

        if mode=="inference" and visualize:
            plot_laplacian_matrix((M[0]/len(data_loader)), plot_file=os.path.join(vis_path, "lap_err.png"))
            plot_laplacian_matrix((M[1]/len(data_loader)), plot_file=os.path.join(vis_path, "lap_est_err.png"), use_est_depth=True)
            plot_laplacian_matrix((M[2]/len(data_loader)), plot_file=os.path.join(vis_path, "lap_count.png"), count=True)
            plot_laplacian_matrix((M[3]/len(data_loader)), plot_file=os.path.join(vis_path, "lap_est_count.png"), use_est_depth=True, count=True)

