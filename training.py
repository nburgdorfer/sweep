import os,sys

from cvt.common import set_random_seed
from src.config import load_config, load_scene_list, get_argparser

parser = get_argparser()
ARGS = parser.parse_args()

#### Select Pipeline ####
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
cfg["mode"] = "training"
cfg["dataset"] = ARGS.dataset
set_random_seed(cfg["seed"])

#### Load Scene Lists ####
ts = load_scene_list(os.path.join(ARGS.config_path, "scene_lists", "training.txt"))
vs = load_scene_list(os.path.join(ARGS.config_path, "scene_lists", "validation.txt"))

#### TRAINING ####
pipeline = Pipeline(cfg, ARGS.config_path, ARGS.log_path, model_name=ARGS.model, training_scenes=ts, validation_scenes=vs)
pipeline.training()

