# Python libraries
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cvtkit.common import to_gpu
from cvtkit.visualization import visualize_mvs

## Custom libraries
from src.pipelines.BasePipeline import BasePipeline
from src.evaluation.eval_2d import depth_acc

# MVSNet Network
from src.networks.MVSNet import Network

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

    def get_network(self):
        return Network(self.cfg).to(self.device)

    def compute_loss(self, data, output):
        loss = {}

        final_depth = output["final_depth"]
        target_depth = data["target_depth"]

        assert (
            final_depth.shape == target_depth.shape
        ), f"Target depth shape ({target_depth.shape}) and Estimate depth shape ({final_depth.shape}) do not match."

        mask = torch.where(target_depth > self.cfg["camera"]["near"], 1.0, 0.0)
        depth_error = (
            F.smooth_l1_loss(final_depth, target_depth, reduction="none") * mask
        )
        depth_error = depth_error.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))

        loss["depth"] = depth_error.mean()
        loss["total"] = loss["depth"]

        return loss

    def compute_stats(self, data, output):
        with torch.set_grad_enabled(
            (torch.is_grad_enabled and not torch.is_inference_mode_enabled)
        ):
            mae, acc = depth_acc(output["final_depth"][0], data["target_depth"][0])
        stats = {"mae": mae, "acc": acc}
        return stats
    
    def update_logger(self, num_samples: int, epoch: int, batch_ind: int, mode: str, log_info: dict[str, float]) -> None:
        """Updates the Summary Writer logger with provided log info values.

        Args:
            num_samples: The number of total batched samples per epoch.
            epoch: The current epoch.
            batch_ind: The current batch index.
            mode: The running mode for the network (usually 'training', 'validation', 'inference').
            log_info: The dictionary of values to log, indexed by a name qualifier.
        """
        iteration = (num_samples * (epoch)) + batch_ind

        for key, val in log_info.items():
            self.logger.add_scalar(
                f"{mode} - {key}",
                val,
                iteration,
            )

    def training(self, epoch):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        self.model.train()
        num_samples = len(self.training_data_loader)
        
        sums = {"loss": 0.0, "mae": 0.0, "acc": 0.0}
        with tqdm(self.training_data_loader, desc=f"MVSNet training - Epoch {epoch}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)

                # Run network forward pass
                output = self.model(data)
                data["target_depth"] = F.interpolate(
                    data["target_depth"], (output["final_depth"].shape[2], output["final_depth"].shape[3]), mode="nearest"
                )

                # Compute loss
                loss = self.compute_loss(data, output)

                # Compute backward pass
                self.optimizer.zero_grad(set_to_none=True)
                loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["optimizer"]["grad_clip"]
                )
                self.optimizer.step()

                # Compute output statistics
                stats = self.compute_stats(data, output)

                # Update progress bar
                max_mem = torch.cuda.max_memory_allocated(device=self.device)
                max_mem = float(max_mem / 1.073742e9)
                log_info = {
                    "loss": float(loss["total"].detach().cpu().item()),
                    "mae": float(stats["mae"].detach().cpu().item()),
                    "acc": float(stats["acc"].detach().cpu().item()) * 100,
                    "max_memory": float(max_mem),
                }
                sums["loss"] += log_info["loss"]
                sums["mae"] += log_info["mae"]
                sums["acc"] += log_info["acc"]
                loader.set_postfix(
                    loss=f"{(sums['loss']/(batch_ind+1)):6.2f}",
                    mae=f"{(sums['mae']/(batch_ind+1)):6.2f}",
                    acc_1cm=f"{(sums['acc']/(batch_ind+1)):3.2f}%",
                    max_memory=f"{(log_info["max_memory"]):2.3f}",
                )

                ## Log loss and statistics
                self.update_logger(num_samples, epoch, batch_ind, "training", log_info)

                ## Visualization
                if self.cfg["training"]["visualize"] and batch_ind % self.cfg["training"]["vis_freq"] == 0:
                    visualize_mvs(
                        data,
                        output,
                        batch_ind,
                        self.log_vis,
                        self.cfg["visualization"]["max_depth_error"],
                        mode="training",
                        epoch=epoch,
                    )
            
                del loss
                del output
                del data
                del stats
                torch.cuda.empty_cache()

    def validation(self, epoch):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        self.model.eval()
        num_samples = len(self.validation_data_loader)
        
        sums = {"loss": 0.0, "mae": 0.0, "acc": 0.0}
        with tqdm(self.validation_data_loader, desc=f"MVSNet validation - Epoch {epoch}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                to_gpu(data, self.device)

                # Run network forward pass
                output = self.model(data)

                # Compute loss
                loss = self.compute_loss(data, output)

                # Compute output statistics
                stats = self.compute_stats(data, output)

                # Update progress bar
                max_mem = torch.cuda.max_memory_allocated(device=self.device)
                max_mem = float(max_mem / 1.073742e9)
                log_info = {
                    "loss": float(loss["total"].detach().cpu().item()),
                    "mae": float(stats["mae"].detach().cpu().item()),
                    "acc": float(stats["acc"].detach().cpu().item()) * 100,
                    "max_memory": float(max_mem),
                }
                sums["loss"] += log_info["loss"]
                sums["mae"] += log_info["mae"]
                sums["acc"] += log_info["acc"]
                loader.set_postfix(
                    loss=f"{(sums['loss']/(batch_ind+1)):6.2f}",
                    mae=f"{(sums['mae']/(batch_ind+1)):6.2f}",
                    acc_1cm=f"{(sums['acc']/(batch_ind+1))*100:3.2f}%",
                    max_memory=f"{(log_info["max_memory"]):2.3f}",
                )

                ## Log loss and statistics
                self.update_logger(num_samples, epoch, batch_ind, "validation", log_info)

                ## Visualization
                if self.cfg["training"]["visualize"] and batch_ind % self.cfg["training"]["vis_freq"] == 0:
                    visualize_mvs(
                        data,
                        output,
                        batch_ind,
                        self.log_vis,
                        self.cfg["visualization"]["max_depth_error"],
                        mode="validation",
                        epoch=epoch,
                    )

    def inference(self):
        self.model.eval()
        with tqdm(self.inference_data_loader, desc=f"MVSNet inference", unit="batch") as loader:
            for data in loader:
                to_gpu(data, self.device)
                output = self.model(data)
                self.save_output(data, output, int(data["ref_id"][0]))