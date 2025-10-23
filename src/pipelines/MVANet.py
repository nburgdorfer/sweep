# Python libraries
import torch
from tqdm import tqdm
import cv2

from cvtkit.common import to_gpu, build_labels
from cvtkit.geometry import project_depth_map
from cvtkit.visualization import visualize_mvs
from cvtkit.metrics import RMSE, chamfer_accuracy, chamfer_completeness

## Custom libraries
from src.pipelines.BasePipeline import BasePipeline
from src.evaluation.eval_2d import depth_acc
from src.evaluation.visualization import write_ply

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
        self.stage_intervals = self.cfg["model"]["stage_intervals"]
        self.stage_training = self.cfg["model"]["stage_training"]
        self.current_resolution = (
            len(self.stage_intervals) if self.stage_training else 0
        )

        self.ply_index = 0

    def get_network(self):
        return Network(self.cfg).to(self.device)

    # def compute_loss_MVS(self, data, output, resolution_stage, final_iteration):
    #     loss = {}
    #     target_depth = data["target_depths"][resolution_stage]
    #     target_labels, mask = build_labels(target_depth, output["hypotheses"])
    #     cost_volume = output["cost_volume"]  # [B, C, H, W]

    #     cost_volume = cost_volume.permute(0, 2, 3, 1)  # [B, H, W, C]
    #     cost_volume = cost_volume[mask > 0]  # [B*H*W, C]
    #     target_labels = target_labels[mask > 0]  # [B*H*W]

    #     error = F.cross_entropy(cost_volume, target_labels, reduction="mean")

    #     # # compute depth rmse loss on final resolution depth map
    #     # if final_iteration:
    #     #     final_depth = output["final_depth"]
    #     #     rmse = RMSE(final_depth, target_depth, mask=torch.where(target_depth>0, 1.0, 0.0))
    #     #     rmse = rmse * self.cfg["loss"]["rmse_weight"]
    #     #     error += rmse

    #     loss["total"] = error * self.loss_weights[resolution_stage]
    #     loss["cov_percent"] = mask.sum() / (
    #         torch.where(target_depth > 0, 1, 0).sum() + 1e-10
    #     )
    #     return loss

    def compute_loss_chamfer(self, output_depth_maps: list[torch.Tensor], output_confidence_maps, data, acc_th=20.0):
        loss = {}
        
        num_views = len(output_depth_maps)

        target_points = data["target_points"]

        # project depth maps
        est_points = None
        # target_points = None
        for i in range(num_views):
            dmap = output_depth_maps[i][0,0].detach().cpu().numpy()
            dmap = (dmap-dmap.min()) / (dmap.max()-dmap.min()+1e-10)
            cv2.imwrite(f"plys/{i:03d}_depth_{self.ply_index:06d}.png", dmap*255)

            gt_mask = torch.where(data["all_target_depths"][:,i].squeeze(1) > 0, 1, 0).to(torch.bool)
            est_points_i = project_depth_map(output_depth_maps[i].squeeze(1), data["extrinsics"][:,i], data["K"], mask=gt_mask)
            if est_points is None:
                est_points = est_points_i
            else:
                est_points = torch.cat((est_points, est_points_i), dim=1)

            # target_points_i = project_depth_map(data["all_target_depths"][:,i].squeeze(1), data["extrinsics"][:,i], data["K"])
            # if target_points is None:
            #     target_points = target_points_i
            # else:
            #     target_points = torch.cat((target_points, target_points_i), dim=1)

        assert est_points is not None
        assert target_points is not None

        # accuracy
        accuracy, _ = chamfer_accuracy(est_points, target_points)
        accuracy = torch.norm(accuracy, dim=2)  
        
        # # completenesst_p
        completeness, _ = chamfer_completeness(est_points, target_points)
        completeness = torch.norm(completeness, dim=2)

        # mask out points with far away
        accuracy = accuracy * torch.where(accuracy < acc_th, 1.0, 0.0)

        # ##### VIS #####
        # write_ply(f"plys/acc_{self.ply_index:08d}.ply", est_points[0], accuracy[0])
        # write_ply(f"plys/comp_{self.ply_index:08d}.ply", target_points[0], completeness[0])
        # # write_ply(f"plys/nearest_target_{self.ply_index:08d}.ply", ret_target_points)
        # # write_ply(f"plys/nearest_est_{self.ply_index:08d}.ply", ret_est_points)
        # self.ply_index += 1
        # ##### VIS #####

        loss["total"] = accuracy.mean() + completeness.mean()
        loss["acc"] = accuracy.mean()
        loss["comp"] = completeness.mean()
        
        return loss

    

    def run(self, mode, epoch):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        if mode == "inference":
            self.model.eval()
            visualize = self.cfg["inference"]["visualize"]
            vis_freq = self.cfg["inference"]["vis_freq"]
            vis_path = self.vis_path
            data_loader = self.inference_data_loader
            dataset = self.inference_dataset
            title_suffix = ""
        else:
            if self.current_resolution == 0:
                self.stage_training = False

            if self.stage_training:
                if epoch >= self.stage_intervals[0]:
                    self.current_resolution -= 1
                    self.stage_intervals.pop(0)

            if mode == "training":
                self.model.train()
                data_loader = self.training_data_loader
                dataset = self.training_dataset
            else:
                self.model.eval()
                data_loader = self.validation_data_loader
                dataset = self.validation_dataset
            visualize = self.cfg["training"]["visualize"]
            vis_freq = self.cfg["training"]["vis_freq"]
            vis_path = self.log_vis
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
                    for iteration, resolution_stage in enumerate(
                        self.resolution_stages
                    ):
                        if (
                            self.stage_training
                            and (resolution_stage < self.current_resolution)
                            and (mode != "inference")
                        ):
                            output_depth_maps.append(output["final_depth"])
                            output_confidence_maps.append(torch.clip(confidence.div_(iteration+1), 0.0, 1.0))
                            break

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
                            data["hypotheses"][iteration + 1] = output[
                                "next_hypotheses"
                            ]
                        if iteration == num_stages-1:
                            output_depth_maps.append(output["final_depth"])
                            output_confidence_maps.append(torch.clip(confidence.div_(iteration+1), 0.0, 1.0))
                
                # clean up data
                del data["image_features"]
                del data["hypotheses"]
                torch.cuda.empty_cache()

                if mode != "inference":
                    loss = self.compute_loss_chamfer(output_depth_maps, output_confidence_maps, data)

                    # Compute backward pass
                    if mode != "validation":
                        loss["total"].backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg["optimizer"]["grad_clip"],
                        )
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

                
            
                    del loss
                    del output_depth_maps
                    del output_confidence_maps
                    del data
                    torch.cuda.empty_cache()
