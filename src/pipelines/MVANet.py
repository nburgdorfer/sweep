# Python libraries
import torch
from tqdm import tqdm
import cv2
import numpy as np
import sys

from torchvision.transforms import Resize, InterpolationMode

from cvtkit.common import to_gpu
from cvtkit.geometry import project_depth_map
from cvtkit.metrics import chamfer_accuracy, chamfer_completeness
from cvtkit.camera import scale_intrinsics

## Custom libraries
from src.pipelines.BasePipeline import BasePipeline
from src.evaluation.visualization import write_ply
from src.components.refiners import BasicRefiner

# GBiNet Network
from src.networks.MVANet import Network

class Pipeline(BasePipeline):
    def __init__(
        self,
        cfg,
        log_path,
        model_name,
        training_scenes=[],
        validation_scenes=[],
        inference_scene=[],
    ):
        super(Pipeline, self).__init__(
            cfg,
            log_path,
            model_name,
            training_scenes,
            validation_scenes,
            inference_scene,
        )
        self.resolution_stages = self.cfg["model"]["resolution_stages"]
        self.ply_index = 0
        self.loss_scale = 0.25
        if self.mode == "training":
            self.resize = Resize(
                    size=(int(self.training_dataset.H * self.loss_scale), int(self.training_dataset.W * self.loss_scale)),
                    interpolation=InterpolationMode.NEAREST,
                )
        else:
            self.resize = Resize(
                    size=(int(self.inference_dataset.H * self.loss_scale), int(self.inference_dataset.W * self.loss_scale)),
                    interpolation=InterpolationMode.NEAREST,
                )


    def get_network(self):
        return (Network(self.cfg).to(self.device), BasicRefiner(in_channels=4, c=8).to(self.device))

    def compute_loss(self, output_depth_maps: list[torch.Tensor], output_confidence_maps, data, scale=0.25):
        loss = {}
        
        num_views = len(output_depth_maps)

        # project depth maps
        est_points = None
        target_points = None
        for i in range(num_views):
            est_depth_map = self.resize(output_depth_maps[i].squeeze(1))
            target_depth_map = self.resize(data["all_target_depths"][:,i].squeeze(1))
            K = scale_intrinsics(data["K"], scale=self.loss_scale)

            # ##### VIS #####
            # dmap = est_depth_map[0].detach().cpu().numpy()
            # dmap = (dmap-dmap.min()) / (dmap.max()-dmap.min()+1e-10)
            # cv2.imwrite(f"plys/{self.ply_index:06d}_{i:03d}_depth.png", dmap*255)
            # tdmap = target_depth_map[0].detach().cpu().numpy()
            # tdmap = (tdmap-tdmap.min()) / (tdmap.max()-tdmap.min()+1e-10)
            # cv2.imwrite(f"plys/{self.ply_index:06d}_{i:03d}_tdepth.png", tdmap*255)
            # ##### VIS #####
            
            gt_mask = torch.where(target_depth_map > 0, 1, 0).to(torch.bool)
            est_points_i = project_depth_map(est_depth_map, data["extrinsics"][:,i], K, mask=gt_mask)
            if est_points is None:
                est_points = est_points_i
            else:
                est_points = torch.cat((est_points, est_points_i), dim=1)

            target_points_i = project_depth_map(target_depth_map, data["extrinsics"][:,i], K)
            if target_points is None:
                target_points = target_points_i
            else:
                target_points = torch.cat((target_points, target_points_i), dim=1)

        assert est_points is not None
        assert target_points is not None

        # accuracy
        accuracy, _ = chamfer_accuracy(est_points, target_points)
        
        # completenesst
        completeness, _ = chamfer_completeness(est_points, target_points)

        # ##### VIS #####
        # write_ply(f"plys/acc_{self.ply_index:08d}.ply", est_points[0], accuracy[0])
        # write_ply(f"plys/comp_{self.ply_index:08d}.ply", target_points[0], completeness[0])
        # # write_ply(f"plys/nearest_target_{self.ply_index:08d}.ply", ret_target_points)
        # # write_ply(f"plys/nearest_est_{self.ply_index:08d}.ply", ret_est_points)
        # self.ply_index += 1
        # ##### VIS #####
        # sys.exit()
        
        loss["total"] = accuracy.mean()# + completeness.mean()
        loss["acc"] = accuracy.mean()
        loss["comp"] = torch.tensor(0.0) #completeness.mean()
        
        return loss

    def run(self, mode, epoch):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        if mode == "inference":
            self.model.eval()
            data_loader = self.inference_data_loader
            dataset = self.inference_dataset
            title_suffix = ""
        else:
            if mode == "training":
                self.model.train()
                data_loader = self.training_data_loader
                dataset = self.training_dataset
            else:
                self.model.eval()
                data_loader = self.validation_data_loader
                dataset = self.validation_dataset
            title_suffix = f" - Epoch {epoch}"
        sums = {"loss": 0.0, "acc": 0.0, "comp": 0.0}

        with tqdm(
            data_loader, desc=f"MVANet {mode}{title_suffix}", unit="batch"
        ) as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)

                num_views = data["images"].shape[1]
                output_depth_maps = []
                output_confidence_maps = []
                loss = {}
                for reference_index in range(num_views):
                    # build image features
                    data["hypotheses"] = [None] * (len(self.resolution_stages))
                    data["image_features"] = None

                    confidence = torch.zeros(
                        (self.batch_size, 1, dataset.H, dataset.W),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    output = {}
                    num_stages = len(self.resolution_stages)
                    for iteration, resolution_stage in enumerate(self.resolution_stages):
                        # Run network forward pass
                        output = self.model(
                            data,
                            resolution_stage=resolution_stage,
                            iteration=iteration,
                            reference_index=reference_index,
                        )
                        confidence += output["confidence"].detach()

                        if iteration == 0:
                            data["hypotheses"][iteration] = output["hypotheses"]
                        if iteration < num_stages - 1:
                            data["hypotheses"][iteration + 1] = output["next_hypotheses"]
                        if iteration == num_stages-1:
                            # Depth Refinement    
                            ref_image =  data["images"][:,0]
                            output["final_depth"] = self.refiner(ref_image, output["final_depth"])

                            output_depth_maps.append(output["final_depth"])
                            output_confidence_maps.append(torch.clip(confidence.div_(iteration+1), 0.0, 1.0))
                
                # clean up data
                del data["image_features"]
                del data["hypotheses"]
                torch.cuda.empty_cache()

                if mode != "inference":
                    loss = self.compute_loss(output_depth_maps, output_confidence_maps, data)

                    # Compute backward pass
                    if mode != "validation":
                        loss["total"].backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg["optimizer"]["grad_clip"],
                        )

                        gradients = []
                        for p in self.model.parameters():
                            if p.grad is not None:
                                gradients.append(float(torch.norm(p.grad)))
                        gradients = np.asarray(gradients)

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Update progress bar
                    sums["loss"] += float(loss["total"].detach().cpu().item())
                    sums["acc"] += float(loss["acc"].detach().cpu().item())
                    sums["comp"] += float(loss["comp"].detach().cpu().item())
                    max_mem = torch.cuda.max_memory_allocated(device=self.device)
                    max_mem = float(max_mem / 1.073742e9)
                    loader.set_postfix(
                        loss=f"{(sums['loss']/(batch_ind+1)):6.2f}",
                        acc=f"{(sums['acc']/(batch_ind+1)):6.2f}",
                        comp=f"{(sums['comp']/(batch_ind+1)):6.2f}",
                        max_memory=f"{(max_mem):2.3f}",
                    )

                    ## Log loss and statistics
                    iteration = (len(loader) * (epoch)) + batch_ind
                    self.logger.add_scalar(
                        f"{mode} - Loss",
                        float(loss["total"].detach().cpu().item()),
                        iteration,
                    )
                    self.logger.add_scalar(
                        f"{mode} - Accuracy",
                        float(loss["acc"].detach().cpu().item()),
                        iteration,
                    )
                    self.logger.add_scalar(
                        f"{mode} - Completeness",
                        float(loss["comp"].detach().cpu().item()),
                        iteration,
                    )
                    self.logger.add_scalar(
                        f"{mode} - Max Memory", float(max_mem), iteration
                    )
                    self.logger.add_scalar(
                        f"{mode} - Max Gradient", float(gradients.max()), iteration
                    )
                    self.logger.add_scalar(
                        f"{mode} - Min Gradient", float(gradients.min()), iteration
                    )
                    self.logger.add_scalar(
                        f"{mode} - Mean Gradient", float(gradients.mean()), iteration
                    )

                    del loss
                    del output_depth_maps
                    del output_confidence_maps
                    del data
                    torch.cuda.empty_cache()
