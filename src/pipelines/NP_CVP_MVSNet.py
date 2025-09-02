# Python libraries
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cvtkit.common import to_gpu
from cvtkit.geometry import edge_mask
from cvtkit.visualization import visualize_mvs

## Custom libraries
from src.pipelines.BasePipeline import BasePipeline
from src.evaluation.eval_2d import depth_acc

# NP-CVP-MVSNet Network
from src.networks.NP_CVP_MVSNet import Network

class Pipeline(BasePipeline):
    def __init__(self, cfg, log_path, model_name, training_scenes=[], validation_scenes=[], inference_scene=[]):
            super(Pipeline, self).__init__(cfg, log_path, model_name, training_scenes, validation_scenes, inference_scene)

    def get_network(self):
        return Network(self.cfg, self.device).to(self.device)

    def compute_loss(self, data, output):
        loss = {}
        losses = []
        loss_level_weights = self.cfg["loss"]["weights"]

        target_depth = data["target_depth"]
        hypos = output["hypos"]
        intervals = output["intervals"]
        prob_grids = output["prob_grids"]
        final_depth = output["final_depth"]

        batch_size, _, height, width = target_depth.shape
        near_depth = torch.ones((batch_size)).to(target_depth) * self.cfg["camera"]["near"]
        far_depth = torch.ones((batch_size)).to(target_depth) * self.cfg["camera"]["far"]

        # Calculate edge mask
        high_frequency_mask = edge_mask(target_depth, near_depth, far_depth)

        resolution_levels = len(self.cfg["model"]["feature_channels"])
        for level in range(resolution_levels):
            if level == resolution_levels-1:
                B,_,D,H,W = prob_grids[level].shape
                mask = (-F.max_pool2d(-target_depth,kernel_size=5,stride=1,padding=2))>near_depth[:,None,None,None]

                # depth loss
                assert(final_depth.shape == target_depth.shape)

                tmp_loss = F.smooth_l1_loss(final_depth[mask], target_depth[mask], reduction='none')
                tmp_high_frequency_mask = high_frequency_mask[mask]
                tmp_high_frequency_weight = tmp_high_frequency_mask.float().mean()
                weight = (1-tmp_high_frequency_weight)*tmp_high_frequency_mask + (tmp_high_frequency_weight)*(1-tmp_high_frequency_mask)
                tmp_loss *= weight
                tmp_loss *= tmp_high_frequency_mask
                tmp_loss *= 0.1
                losses.append(tmp_loss.mean())
            else:
                B,_,D,H,W = prob_grids[level].shape

                # Create gt labels
                unfold_kernel_size = int(2**(resolution_levels-level-1))
                assert unfold_kernel_size%2 == 0 or unfold_kernel_size == 1
                unfolded_patch_depth = F.unfold(target_depth,unfold_kernel_size,dilation=1,padding=0,stride=unfold_kernel_size)
                unfolded_patch_depth = unfolded_patch_depth.reshape(B,1,unfold_kernel_size**2,H,W)
                # valid gt depth mask
                mask = (unfolded_patch_depth>near_depth.view((B,1,1,1,1))).all(dim=2)
                mask *= (unfolded_patch_depth<far_depth.view((B,1,1,1,1))).all(dim=2)
                # Approximate depth distribution from depth observations
                gt_occ_grid = torch.zeros_like(hypos[level])
                for pixel in range(unfolded_patch_depth.shape[2]):
                    selected_depth = unfolded_patch_depth[:,:,pixel]
                    distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                    distance_to_hypo /= intervals[level]
                    mask = distance_to_hypo>1
                    weights = 1-distance_to_hypo
                    weights[mask] = 0
                    gt_occ_grid+=weights
                gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
                gt_occ_grid[torch.isnan(gt_occ_grid)] = 0

                covered_mask = gt_occ_grid.sum(dim=2,keepdim=True) > 0
                occ_hypos_count = (gt_occ_grid>0).sum(dim=2,keepdim=True).repeat(1,1,D,1,1)
                edge_weight = occ_hypos_count
                final_mask = mask.unsqueeze(2) * covered_mask

                est = torch.masked_select(prob_grids[level],final_mask)
                gt = torch.masked_select(gt_occ_grid,final_mask)
                tmp_loss = F.binary_cross_entropy(est,gt, reduction="none")
                edge_weight = torch.masked_select(edge_weight,final_mask)
                # Apply edge weight
                tmp_loss = tmp_loss * edge_weight
                # class balance
                num_positive = (gt>0).sum()
                num_negative = (gt==0).sum()
                num_total = gt.shape[0]
                alpha_positive = num_negative/float(num_total)
                alpha_negative = num_positive/float(num_total)
                weight = alpha_positive*(gt>0) + alpha_negative*(gt==0)
                tmp_loss = weight*tmp_loss
                tmp_loss = tmp_loss.mean()
                tmp_loss = loss_level_weights[level]*tmp_loss
                losses.append(tmp_loss)

        loss["depth"] = torch.stack(losses).mean()
        loss["total"] = loss["depth"]
        return loss

    def compute_stats(self, data, output):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            mae, acc = depth_acc(output["final_depth"][0,0], data["target_depth"][0,0])
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
            title_suffix = f"{mode}"
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
            title_suffix = f"{mode} - Epoch {epoch}"
            sums = {
                    "loss": 0.0,
                    "mae": 0.0,
                    "acc": 0.0
                    }


        with tqdm(data_loader, desc=f"NP-CVP-MVSNet {title_suffix}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)

                # Run network forward pass
                output = self.model(data)

                if mode != "inference":
                    # Compute loss
                    loss = self.compute_loss(data, output)

                    # Compute backward pass
                    if mode != "validation":
                        self.optimizer.zero_grad(set_to_none=True)
                        loss["total"].backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["grad_clip"])
                        self.optimizer.step()

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
                    self.save_output(data, output, int(data["ref_id"][0]))

                ## Visualization
                if (visualize and batch_ind % vis_freq == 0):
                    visualize_mvs(data, output, batch_ind, vis_path, self.cfg["visualization"]["max_depth_error"], mode=mode, epoch=epoch)

                if mode != "inference":
                    del loss
                    del output
                    del data
                    del stats
                    torch.cuda.empty_cache()
