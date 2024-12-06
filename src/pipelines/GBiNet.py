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
from cvt.common import to_gpu, laplacian_pyramid, build_labels
from cvt.io import write_pfm
from cvt.geometry import visibility, get_uncovered_mask, edge_mask
from cvt.visualization import visualize_mvs, laplacian_depth_error, laplacian_count, laplacian_uncovered_count, plot_laplacian_matrix

# GBiNet Network
from src.networks.GBiNet import Network

class Pipeline(BasePipeline):
    def __init__(self, cfg, config_path, log_path, model_name, training_scenes=None, validation_scenes=None, inference_scene=None):
        super(Pipeline, self).__init__(cfg, config_path, log_path, model_name, training_scenes, validation_scenes, inference_scene)
        self.stage_ids = self.cfg["model"]["stage_ids"]
        self.loss_weights = self.cfg["loss"]["weights"]
        self.confidence_iterations = self.cfg["model"]["confidence_iterations"]

    def get_network(self):
        return Network(self.cfg).to(self.device)

    def compute_loss(self, data, output, stage_id):
        loss = {}
        target_depth = data["target_depths"][stage_id]
        target_labels, mask = build_labels(target_depth, output["hypotheses"])
        cost_volume = output["cost_volume"]

        b, _, h, w = target_depth.shape

        error = F.cross_entropy(cost_volume, target_labels, reduction='none')
        error *= mask

        loss["total"] = (error.sum(dim=(1,2)) / (mask.sum(dim=(1,2))+1e-10)).mean() * self.loss_weights[stage_id]
        loss["cov_percent"] = (mask.sum() / torch.where(target_depth > 0, 1, 0).sum())
        return loss

    def compute_stats(self, data, output, stage_id):
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
            self.model.eval()
            visualize = self.cfg["inference"]["visualize"]
            vis_freq = self.cfg["inference"]["vis_freq"]
            vis_path = self.vis_path
            data_loader = self.inference_data_loader
            title_suffix = ""
        else:
            if mode == "training":
                self.model.train()
                data_loader = self.training_data_loader
            elif mode == "validation":
                self.model.eval()
                data_loader = self.validation_data_loader
            visualize = self.cfg["training"]["visualize"]
            vis_freq = self.cfg["training"]["vis_freq"]
            vis_path = self.log_vis
            title_suffix = f" - Epoch {epoch}"
            sums = {
                    "loss": 0.0,
                    "cov_percent": 0.0,
                    "mae": 0.0,
                    "acc": 0.0
                    }

        with tqdm(data_loader, desc=f"MVS-Studio {mode}{title_suffix}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)
                
                # build image features
                data["hypotheses"] = [None]*(len(self.stage_ids))

                for iteration, stage_id in enumerate(self.stage_ids):
                    if mode == "training":
                        self.optimizer.zero_grad()

                    if iteration > epoch+1 and mode != "inference":
                        break

                    # Run network forward pass
                    output = self.model(data, stage_id=stage_id, iteration=iteration)

                    if iteration==0:
                        data["hypotheses"][iteration] = output["hypotheses"]
                        confidence = output["confidence"]
                    if iteration < len(self.stage_ids)-1:
                        data["hypotheses"][iteration+1] = output["next_hypotheses"]
                    if iteration <= self.confidence_iterations:
                        confidence += output["confidence"]

                    if mode != "inference":
                        # Compute loss
                        loss = self.compute_loss(data, output, stage_id)

                        # Compute backward pass
                        if mode != "validation":
                            loss["total"].backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["grad_clip"])
                            self.optimizer.step()

                # confidence average
                output["confidence"] = confidence / self.confidence_iterations

                if mode != "inference":
                    # Compute output statistics
                    stats = self.compute_stats(data, output, stage_id)

                    # Update progress bar
                    sums["loss"] += float(loss["total"].detach().cpu().item())
                    sums["cov_percent"] += float(loss["cov_percent"].detach().cpu().item())
                    sums["mae"] += float(stats["mae"].detach().cpu().item())
                    sums["acc"] += float(stats["acc"].detach().cpu().item())
                    max_mem = torch.cuda.max_memory_allocated(device=self.device)
                    max_mem = float(max_mem / 1.073742e9)
                    loader.set_postfix(
                            loss=f"{(sums['loss']/(batch_ind+1)):6.2f}",
                            cover=f"{(sums['cov_percent']/(batch_ind+1))*100:6.2f}%",
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
                    self.save_output(data, output, int(data["ref_id"][0]))

                ## Visualization
                if (visualize and batch_ind % vis_freq == 0):
                    visualize_mvs(data, output, batch_ind, vis_path, self.cfg["visualization"]["max_depth_error"], mode=mode, epoch=epoch)

                if mode != "inference":
                    del loss
                    del output
                    del data
                    if epoch > 5:
                        del stats
                    torch.cuda.empty_cache()
