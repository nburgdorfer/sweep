import numpy as np
import sys
import os
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cvtkit.io import read_pfm
from cvtkit.metrics import MAE

def projection_eval(est_depth_path, est_conf_path, dataset):
    all_depth_files = os.listdir(est_depth_path)

    # get depth map filenames
    est_depth_files = [df for df in all_depth_files if df[8:] == ".pfm" ]
    est_depth_files.sort()
    # get filtered map filenames
    filt_depth_files = [df for df in all_depth_files if df[8:] == "_filt.pfm" ]
    filt_depth_files.sort()
    # get projected map filenames
    proj_depth_files = [df for df in all_depth_files if df[8:] == "_proj.pfm" ]
    proj_depth_files.sort()

    depth_files = np.stack((est_depth_files, filt_depth_files, proj_depth_files), axis=-1)

    # load target depth maps
    target_depths = dataset.get_all_depths()

    perc_better = np.zeros((len(est_depth_files)), dtype=np.float32)
    mae_better = np.zeros((len(est_depth_files)), dtype=np.float32)
    mae_worse = np.zeros((len(est_depth_files)), dtype=np.float32)
    for i, (edf, fdf, pdf) in enumerate(depth_files):
        ref_ind = int(edf[:8])
        est_depth = read_pfm(os.path.join(est_depth_path, edf))
        filt_depth = read_pfm(os.path.join(est_depth_path, fdf))
        proj_depth = read_pfm(os.path.join(est_depth_path, pdf))
        target_depth = target_depths[ref_ind].cpu().numpy()

        gt_mask = np.where(target_depth > 0, 1, 0)
        filt_mask = np.where(filt_depth > 0, 0, 1)
        proj_mask = np.where(proj_depth > 0, 1, 0)
        mask = proj_mask * filt_mask * gt_mask

        # compute dense depth error
        dense_error = (np.abs(est_depth - target_depth) * mask).flatten()
        proj_error = (np.abs(proj_depth - target_depth) * mask).flatten()
        diff = np.abs(dense_error - proj_error)

        better_inds = np.argwhere(proj_error < dense_error)
        worse_inds = np.argwhere(proj_error >= dense_error)

        perc_better[i] = better_inds.shape[0] / (better_inds.shape[0]+worse_inds.shape[0])
        mae_better[i] = diff[better_inds].mean()
        mae_worse[i] = diff[worse_inds].mean()
        
    return perc_better.mean(), mae_better.mean(), mae_worse.mean()

def depth_fscore(est_depth_path, est_conf_path, dataset, th=2.0):
    all_depth_files = os.listdir(est_depth_path)

    # get depth map filenames
    est_depth_files = [df for df in all_depth_files if df[8:] == ".pfm" ]
    est_depth_files.sort()
    # get filtered map filenames
    filt_depth_files = [df for df in all_depth_files if df[8:] == "_filt.pfm" ]
    filt_depth_files.sort()
    # get projected map filenames
    proj_depth_files = [df for df in all_depth_files if df[8:] == "_proj.pfm" ]
    proj_depth_files.sort()

    depth_files = np.stack((est_depth_files, filt_depth_files, proj_depth_files), axis=-1)

    # load target depth maps
    target_depths = dataset.get_all_depths()

    precision = np.zeros((len(est_depth_files),3), dtype=np.float32)
    recall = np.zeros((len(est_depth_files),3), dtype=np.float32)
    fscore = np.zeros((len(est_depth_files),3), dtype=np.float32)
    for i, (edf, fdf, pdf) in enumerate(depth_files):
        ref_ind = int(edf[:8])
        est_depth = read_pfm(os.path.join(est_depth_path, edf))
        filt_depth = read_pfm(os.path.join(est_depth_path, fdf))
        proj_depth = read_pfm(os.path.join(est_depth_path, pdf))
        target_depth = target_depths[ref_ind].cpu().numpy()

        gt_mask = np.where(target_depth > 0, 1, 0)
        filt_mask = np.where(filt_depth > 0, 1, 0) * gt_mask
        proj_mask = np.where(proj_depth > 0, 1, 0) * gt_mask

        # compute dense depth error
        dense_error = np.abs(est_depth - target_depth) * gt_mask
        dense_error = dense_error[gt_mask > 0]
        precision[i,0] = np.argwhere(dense_error <= th).shape[0] / (dense_error.shape[0] + 1e-10)
        recall[i,0] = np.argwhere(dense_error <= th).shape[0] / (dense_error.shape[0] + 1e-10)
        fscore[i,0] = (2*precision[i,0]*recall[i,0]) / (precision[i,0] + recall[i,0])

        # compute filtered depth error
        filt_error = np.abs(filt_depth - target_depth)
        filt_acc = filt_error * filt_mask
        filt_acc = filt_acc[filt_mask > 0]
        precision[i,1] = np.argwhere(filt_acc <= th).shape[0] / (filt_acc.shape[0] + 1e-10)

        filt_comp = filt_error * gt_mask
        filt_comp = filt_comp[gt_mask > 0]
        recall[i,1] = np.argwhere(filt_comp <= th).shape[0] / (filt_comp.shape[0] + 1e-10)
        fscore[i,1] = (2*precision[i,1]*recall[i,1]) / (precision[i,1] + recall[i,1])

        # compute filtered depth error
        proj_error = np.abs(proj_depth - target_depth)
        proj_acc = proj_error * proj_mask
        proj_acc = proj_acc[proj_mask > 0]
        precision[i,2] = np.argwhere(proj_acc <= th).shape[0] / (proj_acc.shape[0] + 1e-10)

        proj_comp = proj_error * gt_mask
        proj_comp = proj_comp[gt_mask > 0]
        recall[i,2] = np.argwhere(proj_comp <= th).shape[0] / (proj_comp.shape[0] + 1e-10)
        fscore[i,2] = (2*precision[i,2]*recall[i,2]) / (precision[i,2] + recall[i,2])
        
    return precision.mean(axis=0), recall.mean(axis=0), fscore.mean(axis=0)


def eval_2d(paths, dataset, vis_path=None, scale=True):
    # threshold values used for depth error pixel percentages
    thresholds = [0.125, 0.25, 0.5, 1.0]
    re_files = 10

    # get depth map filenames
    est_depth_files = os.listdir(paths["depth"])
    est_depth_files = [edf for edf in est_depth_files if edf[-3:] == "pfm" ]
    est_depth_files.sort()

    # load target depth maps
    target_depths = dataset.get_all_depths(scale)

    mae = np.zeros((len(est_depth_files)), dtype=np.float32)
    auc = np.zeros((len(est_depth_files)), dtype=np.float32)
    percentages = np.zeros((len(est_depth_files), len(thresholds)), dtype=np.float32)
    maps = {}
    for i, edf in enumerate(est_depth_files):
        ref_ind = int(edf[:8])
        maps["est_depth"] = torch.tensor(read_pfm(os.path.join(paths["depth"], edf)))
        maps["est_conf"] = torch.tensor(read_pfm(os.path.join(paths["confidence"], f"{ref_ind:08d}.pfm")))
        maps["target_depth"] = torch.tensor(target_depths[ref_ind])[0]
        mask = torch.where(maps["target_depth"] > 0, 1, 0)
        
        # compute MAE
        mae[i] = float(MAE(maps["est_depth"], maps["target_depth"], reduction_dims=(0,1), mask=mask).item())

        # compute AUC
        auc[i] = float(auc_score(maps, re_files, ref_ind, mask=mask, vis_path=vis_path))

        # compute pixel percentages
        percs = depth_percentages(maps["est_depth"], maps["target_depth"], thresholds=thresholds, mask=mask)
        percentages[i] = percs

    mae = mae.mean()
    auc = auc.mean()
    percentages = percentages.mean(axis=0)
    percentages *= 100
    print(f"MAE: {mae:6.3f}")
    print(f"AUC: {auc:6.3f}")
    print(f"Percentages: {percentages[0]:3.2f}%  |  {percentages[1]:3.2f}%  |  {percentages[2]:3.2f}%  |  {percentages[3]:3.2f}%")

    return mae, auc, percentages

def auc_score(maps, re_files, ref_ind, mask, vis_path):
    # grab valid reprojection error values
    pixel_count = int(torch.sum(mask).item())

    est_depth = maps["est_depth"]
    est_conf = maps["est_conf"]
    target_depth = maps["target_depth"]

    inds = torch.where(mask==1)
    target_depth = target_depth[inds].cpu().numpy()
    est_depth = est_depth[inds].cpu().numpy()
    est_conf = est_conf[inds].cpu().numpy()

    # flatten to 1D tensor
    target_depth = target_depth.flatten()
    est_depth = est_depth.flatten()
    est_conf = est_conf.flatten()

    # compute error
    error = np.abs(est_depth - target_depth)
    
    # sort orcale curves by error
    oracle_indices = np.argsort(error)
    oracle_error = np.take(error, indices=oracle_indices, axis=0)

    # sort all tensors by confidence value
    est_indices_conf = np.argsort(est_conf)
    est_indices_conf = est_indices_conf[::-1]
    est_error_conf = np.take(error, indices=est_indices_conf, axis=0)

    # build density vector
    perc = np.array(list(range(5,105,5)))
    density = np.array((perc/100) * (pixel_count), dtype=np.int32)

    oracle_roc = np.zeros(density.shape)
    est_roc_conf = np.zeros(density.shape)
    for i,k in enumerate(density):
        oe = oracle_error[:k]
        ee_conf = est_error_conf[:k]

        if (oe.shape[0] == 0):
            oracle_roc[i] = 0.0
            est_roc_conf[i] = 0.0
        else:
            oracle_roc[i] = np.mean(oe)
            est_roc_conf[i] = np.mean(ee_conf)

    # comput AUC
    oracle_auc = np.trapz(oracle_roc, dx=1)
    est_auc_conf = np.trapz(est_roc_conf, dx=1)

    if(vis_path!=None):
        # plot ROC density errors
        plt.plot(perc, oracle_roc, label="Oracle")
        plt.plot(perc, est_roc_conf, label="Confidence")
        plt.title("ROC Error")
        plt.xlabel("density")
        plt.ylabel("absolte error")
        plt.legend()
        plt.savefig(os.path.join(vis_path,f"roc_{ref_ind:08d}.png"))
        plt.close()

    return est_auc_conf


def depth_percentages(estimate, target, thresholds, mask=None, relative=False):
    assert(estimate.shape==target.shape)
    num_pixels = estimate.flatten().shape[0]

    error = estimate - target
    if relative:
        error /= target
    error = torch.abs(error)

    percs = []
    if mask != None:
        assert(error.shape==mask.shape)
        error *= mask

    for th in thresholds:
        inliers = torch.where(error <= th, 1, 0)
        if mask != None:
            inliers *= mask
            perc = (inliers.sum()) / (mask.sum()+1e-10)
        else:
            perc = inliers.sum() / num_pixels
        percs.append(perc)
    return np.asarray(percs)
    

def target_coverage(data, outputs):
    pass

    target_depth = data["target_depth"]
    loss = []
    near_depth = torch.ones((cfg["training"]["batch_size"])).to(cfg["device"]) * self.cfg["camera"]["near"]
    far_depth = torch.ones((cfg["training"]["batch_size"])).to(cfg["device"]) * self.cfg["camera"]["far"]

    hypos = outputs["hypos"]
    hypo_coords = outputs["hypo_coords"]
    intervals = outputs["intervals"]
    global_probs = outputs["global_probs"]
    prob_grids = outputs["prob_grids"]

    # Calculate edge mask
    down_gt = F.interpolate(target_depth.unsqueeze(1),scale_factor=0.5,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    down_up_gt = F.interpolate(down_gt,scale_factor=2,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    res = torch.abs(target_depth.unsqueeze(1)-down_up_gt)
    high_frequency_mask = res>(0.001*(far_depth-near_depth)[:,None,None,None])
    valid_gt_mask = (-F.max_pool2d(-target_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>near_depth[:,None,None,None]
    high_frequency_mask = high_frequency_mask * valid_gt_mask

    for level in reversed(range(len(cfg["model"]["gwc_groups"]))):
        if level ==0:
            # Apply softargmax depth regression for subpixel depth estimation on final level.
            B,_,D,H,W = prob_grids[level].shape

            final_prob = prob_grids[level]
            final_hypo = hypos[level]
            regressed_depth = torch.sum(final_prob*final_hypo,dim=2)
            gt_depth = target_depth.unsqueeze(1)

            mask = (-F.max_pool2d(-target_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>near_depth[:,None,None,None]
            tmp_loss = F.smooth_l1_loss(regressed_depth[mask], gt_depth[mask], reduction='none')

            tmp_high_frequency_mask = high_frequency_mask[mask]
            tmp_high_frequency_weight = tmp_high_frequency_mask.float().mean()
            weight = (1-tmp_high_frequency_weight)*tmp_high_frequency_mask + (tmp_high_frequency_weight)*(~tmp_high_frequency_mask)
            tmp_loss *= weight
            tmp_loss *= 0.1
            loss.append(tmp_loss.mean())
            continue

        B,_,D,H,W = prob_grids[level].shape

        # Create gt labels
        unfold_kernel_size = int(2**level)
        assert unfold_kernel_size%2 == 0 or unfold_kernel_size == 1
        unfolded_patch_depth = F.unfold(target_depth.unsqueeze(1),unfold_kernel_size,dilation=1,padding=0,stride=unfold_kernel_size)
        unfolded_patch_depth = unfolded_patch_depth.reshape(B,1,unfold_kernel_size**2,H,W)
        # valid gt depth mask
        mask = (unfolded_patch_depth>near_depth.view((B,1,1,1,1))).all(dim=2)
        mask *= (unfolded_patch_depth<far_depth.view((B,1,1,1,1))).all(dim=2)
        # Approximate depth distribution from depth observations
        gt_occ_grid = torch.zeros_like(hypos[level])
        if self.cfg["loss"]["gt_prob_mode"] == "hard":
            for pixel in range(unfolded_patch_depth.shape[2]):
                selected_depth = unfolded_patch_depth[:,:,pixel]
                distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                occupied_mask = distance_to_hypo<=(intervals[level]/2)
                gt_occ_grid[occupied_mask]+=1
            gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
            gt_occ_grid[torch.isnan(gt_occ_grid)] = 0
        elif self.cfg["loss"]["gt_prob_mode"] == "soft":
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


        if self.cfg["loss"]["func"]=="BCE":
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
            loss.append(tmp_loss)
        elif self.cfg["loss"]["func"]=="KL":
            # KL loss
            est = torch.masked_select(prob_grids[level],final_mask)
            gt = torch.masked_select(gt_occ_grid,final_mask)
            tmp_loss = F.kl_div(est.log(),gt, reduction="none", log_target=False)
            edge_weight = torch.masked_select(edge_weight,final_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            tmp_loss = tmp_loss.mean()
            tmp_loss = loss_level_weights[level]*tmp_loss
            loss.append(tmp_loss)

    loss = torch.stack(loss).mean()
    return loss

def depth_acc(est_depth, gt_depth, th=1.0):
    assert(est_depth.shape == gt_depth.shape)
    abs_error = torch.abs(est_depth - gt_depth)
    valid_pixels = torch.where(gt_depth > 0, 1, 0)
    acc_pixels = torch.where(abs_error <= th, 1, 0)

    abs_error *= valid_pixels
    acc_pixels *= valid_pixels

    mae = abs_error.sum() / (valid_pixels.sum()+1e-5)
    acc = acc_pixels.sum() / (valid_pixels.sum()+1e-5)

    return mae, acc
