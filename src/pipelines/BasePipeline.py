# Python libraries
import cv2
import os
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cvtkit.common import parameters_count
from cvtkit.io import write_pfm, load_ckpt, save_ckpt, write_cam_sfm

## Custom libraries
from src.config import save_config
from src.datasets.BaseDataset import build_dataset

class BasePipeline():
    def __init__(self,
                 cfg: dict[str, Any],
                 log_path: str,
                 model_name: str,
                 training_scenes: list[str],
                 validation_scenes: list[str],
                 inference_scene: list[str]):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.mode = self.cfg["mode"]
        self.training_scenes = training_scenes
        self.validation_scenes = validation_scenes
        self.inference_scene = inference_scene
        self.model_name = model_name
        self.depth_range = self.cfg["camera"]["far"] - self.cfg["camera"]["near"]

        # build the data loaders
        self.build_dataset()
        self.build_model()

        # set data paths
        self.log_path = log_path
        self.ckpt_path = os.path.join(self.log_path, "ckpts")
        self.log_vis = os.path.join(self.log_path, "visuals")
        self.final_ckpt_file = os.path.join(self.ckpt_path, "model.pt") 
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.log_vis, exist_ok=True)
        self.logger = SummaryWriter(log_dir=self.log_path)

        if (self.mode=="inference"):
            # set data paths
            self.data_path = os.path.join(self.cfg["data_path"], self.inference_scene[0])
            self.output_path = os.path.join(self.cfg["output_path"], self.inference_scene[0])
            self.camera_path = os.path.join(self.output_path, "camera")
            self.depth_path = os.path.join(self.output_path, "depth")
            self.conf_path = os.path.join(self.output_path, "conf")
            self.image_path = os.path.join(self.output_path, "image")
            self.vis_path = os.path.join(self.output_path, "visuals")
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.camera_path, exist_ok=True)
            os.makedirs(self.depth_path, exist_ok=True)
            os.makedirs(self.conf_path, exist_ok=True)
            os.makedirs(self.image_path, exist_ok=True)
            os.makedirs(self.vis_path, exist_ok=True)
            self.batch_size = 1
        else:
            self.batch_size = self.cfg["training"]["batch_size"]
            self.build_optimizer()
            self.build_scheduler()

        # log current configuration used
        save_config(self.log_path, self.cfg)

    def get_network(self):
        raise NotImplementedError()

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def compute_stats(self, *args, **kwargs):
        raise NotImplementedError()

    def save_output(self, data, output, sample_ind):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            # save confidence map
            conf_map = output["confidence"][0,0].detach().cpu().numpy()
            conf_filename = os.path.join(self.conf_path, f"{sample_ind:08d}.pfm")
            write_pfm(conf_filename, conf_map)
            conf_map = output["confidence"][0,0].detach().cpu().numpy()
            os.makedirs(os.path.join(self.conf_path, "disp"), exist_ok=True)
            conf_filename = os.path.join(self.conf_path, "disp", f"{sample_ind:08d}.png")
            cv2.imwrite(conf_filename, (conf_map * 255))
            # save depth map
            depth_map = output["final_depth"][0,0].detach().cpu().numpy()
            depth_filename = os.path.join(self.depth_path, f"{sample_ind:08d}.pfm")
            write_pfm(depth_filename, depth_map)
            depth_map = (output["final_depth"][0,0].detach().cpu().numpy() - self.cfg["camera"]["near"]) / (self.cfg["camera"]["far"]-self.cfg["camera"]["near"])
            os.makedirs(os.path.join(self.depth_path, "disp"), exist_ok=True)
            depth_filename = os.path.join(self.depth_path, "disp", f"{sample_ind:08d}.png")
            cv2.imwrite(depth_filename, (depth_map * 255))
            # save image
            ref_image = (torch.movedim(data["images"][0,0], (0,1,2), (2,0,1)).detach().cpu().numpy()+0.5) * 255
            img_filename = os.path.join(self.image_path, f"{sample_ind:08d}.png")
            cv2.imwrite(img_filename, ref_image[:,:,::-1])
            # save camera
            intrinsics = data["K"][0].detach().cpu().numpy()
            extrinsics = data["poses"][0,0].detach().cpu().numpy()
            cam_filename = os.path.join(self.camera_path, f"{sample_ind:08d}_cam.txt")
            write_cam_sfm(cam_filename, intrinsics, extrinsics)

    def build_dataset(self):
        if (self.mode=="training"):
            self.training_dataset = build_dataset(self.cfg, "training", self.training_scenes)
            self.cfg["H"], self.cfg["W"] = self.training_dataset.H, self.training_dataset.W
            self.training_data_loader = DataLoader(self.training_dataset,
                                         self.cfg["training"]["batch_size"],
                                         shuffle=True,
                                         num_workers=self.cfg["num_workers"],
                                         pin_memory=True,
                                         drop_last=True)

            self.validation_dataset = build_dataset(self.cfg, "validation", self.validation_scenes)
            self.validation_data_loader = DataLoader(self.validation_dataset,
                                         self.cfg["training"]["batch_size"],
                                         shuffle=True,
                                         num_workers=self.cfg["num_workers"],
                                         pin_memory=True,
                                         drop_last=True)

        else:
            self.inference_dataset = build_dataset(self.cfg, self.mode, self.inference_scene)
            self.cfg["H"], self.cfg["W"] = self.inference_dataset.H, self.inference_dataset.W
            self.inference_data_loader = DataLoader(self.inference_dataset,
                                         1,
                                         shuffle=False,
                                         num_workers=self.cfg["num_workers"],
                                         pin_memory=True,
                                         drop_last=False)

    def build_model(self):
        self.parameters_to_train = []
        self.model = self.get_network() # get specific network for the requested network
        self.parameters_to_train += list(self.model.parameters())
        parameters_count(self.model, self.model_name)

        if self.mode == "inference":
            print(f"Loading pretrained model '{self.cfg['inference']['ckpt_file']}'.")
            load_ckpt(self.model, self.cfg["inference"]["ckpt_file"])
        else:
            if (self.cfg["training"]["ckpt_file"] != None):
                print(f"Continuing training pretrained model '{self.cfg['training']['ckpt_file']}'.")
                load_ckpt(self.model, self.cfg["training"]["ckpt_file"])

    def build_optimizer(self):
        rate = self.cfg["optimizer"]["learning_rate"]
        self.optimizer = optim.AdamW(self.parameters_to_train, lr=rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    def build_scheduler(self):
        T_max = self.cfg["training"]["epochs"]
        eta_min = self.cfg["optimizer"]["eta_min"]
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=T_max, eta_min=eta_min)
        # T_max = self.cfg["training"]["epochs"]
        # eta_min = self.cfg["optimizer"]["eta_min"]
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=5, T_mult=2)

    def training(self):
        for epoch in range(self.cfg["training"]["epochs"]):
            self.run(mode="training", epoch=epoch)
            with torch.inference_mode():
                self.run(mode="validation", epoch=epoch)

            # save model checkpoint
            ckpt_file = os.path.join(self.ckpt_path, f"ckpt_{epoch:04d}.pt") 
            save_ckpt(self.model, ckpt_file)

            # shuffle scenes in data loaders
            self.training_dataset.shuffle_and_subsample()
            self.validation_dataset.shuffle_and_subsample()
            
            self.lr_scheduler.step(epoch=epoch)
            torch.cuda.empty_cache()

        # save final model checkpoint
        save_ckpt(self.model, self.final_ckpt_file)
        return

    def inference(self):
        with torch.inference_mode():
            self.run(mode="inference", epoch=-1)
        return

    def run(self, mode, epoch):
        raise NotImplementedError()
