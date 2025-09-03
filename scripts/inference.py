import os, sys
import numpy as np

from cvtkit.common import set_random_seed
from cvtkit.visualization.util import print_csv, to_normal
from cvtkit.filtering import consensus_filter

from src.evaluation.eval_2d import eval_2d
from src.evaluation.eval_3d import dtu_point_eval
from src.config import load_config, get_argparser

parser = get_argparser()
ARGS = parser.parse_args()

#### Import Pipeline ####
if ARGS.model == "MVSNet":
    from src.pipelines.MVSNet import Pipeline
elif ARGS.model == "NP_CVP_MVSNet":
    from src.pipelines.NP_CVP_MVSNet import Pipeline
elif ARGS.model == "GBiNet":
    from src.pipelines.GBiNet import Pipeline
else:
    print(f"Error: unknown method {ARGS.model}.")
    sys.exit()


#### Load Configuration ####
cfg = load_config(os.path.join(ARGS.config_path, f"{ARGS.dataset}.yaml"))
cfg["mode"] = "inference"
cfg["dataset"] = ARGS.dataset
set_random_seed(cfg["seed"])


#### Load Scene Lists ####
scene_list = os.path.join(ARGS.config_path, "scene_lists", "inference.txt")
with open(scene_list,'r') as sf:
    scenes = sf.readlines()
    scenes = [s.strip() for s in scenes]


#### INFERENCE ####
avg_mae = np.zeros((len(scenes)))
avg_auc = np.zeros((len(scenes)))
avg_percentages = np.zeros((len(scenes), 4))
avg_acc = np.zeros((len(scenes)))
avg_comp = np.zeros((len(scenes)))
avg_fscore = np.zeros((len(scenes)))

for i, scene in enumerate(scenes):
    print(f"\n----Running MVS on {scene}----")
    pipeline = Pipeline(cfg=cfg, log_path=ARGS.log_path, model_name=ARGS.model, inference_scene=[scene])
    pipeline.inference()

    ####### 2D EVALUATION ####
    print("\n---Evaluating depth maps---")
    paths = {
            "depth": pipeline.depth_path,
            "confidence":pipeline.conf_path,
            "rgb": pipeline.image_path}
    mae, auc, percentages = eval_2d(paths, pipeline.inference_dataset, pipeline.vis_path, scale=False)
    avg_mae[i] = mae
    avg_auc[i] = auc
    avg_percentages[i] = percentages

    #### 3D EVALUATION ####
    print("\n---Evaluating point cloud---")
    consensus_filter(cfg, pipeline.depth_path, pipeline.conf_path, pipeline.image_path, pipeline.output_path, pipeline.inference_dataset, scene)
    to_normal(os.path.join(pipeline.output_path, f"{scene}.ply"), os.path.join(pipeline.output_path, f"{scene}_normal.ply"))
    acc, comp, prec, rec, fscore = dtu_point_eval(cfg, scene, method=ARGS.model)
    avg_acc[i] = acc
    avg_comp[i] = comp
    avg_fscore[i] = fscore

print("\n---MAE list---")
print_csv(avg_mae)

print("\n---AUC list---")
print_csv(avg_auc)

print("\n---F-Score list---")
print_csv(avg_fscore)

avg_mae = avg_mae.mean()
avg_auc = avg_auc.mean()
avg_percentages = avg_percentages.mean(axis=0)
avg_acc = avg_acc.mean()
avg_comp = avg_comp.mean()
avg_fscore = avg_fscore.mean()
print(f"\n---Average---\nMAE: {avg_mae:6.3f}{pipeline.inference_dataset.units}")
print(f"AUC: {avg_auc:6.3f}")
print(f"Percentages: {avg_percentages[0]:3.2f}%  |  {avg_percentages[1]:3.2f}%  |  {avg_percentages[2]:3.2f}%  |  {avg_percentages[3]:3.2f}%")
print(f"ACC: {avg_acc:6.3f}")
print(f"COMP: {avg_comp:6.3f}")
print(f"F-Score: {avg_fscore:6.3f}")
