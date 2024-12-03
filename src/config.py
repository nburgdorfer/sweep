import argparse
import yaml
import os

def get_argparser():
    parser = argparse.ArgumentParser(description='Arguments for running MVS-Studio.')
    parser.add_argument('--config_path', type=str, help='Path to config file.', required=True)
    parser.add_argument('--log_path', type=str, help='Path to log directory.', required=True)
    parser.add_argument('--dataset', type=str, help='Current dataset being used.', required=True, choices=["ScanNet", "Replica", "DTU", "TNT"])
    parser.add_argument('--model', type=str, help='Current model to be used.', required=True, choices=["NP_CVP_MVSNet", "GBiNet", "MVSNet"])
    return parser

def save_config(output_path, cfg):
    with open(os.path.join(output_path, "config.yaml"), 'w') as config_file:
        yaml.dump(cfg, config_file)

def load_config(config_path):
    # load configuration from file itself
    with open(config_path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    if inherit_from is not None:
        cfg = load_config(inherit_from)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def load_bounding_box(cfg, bb_path):
    with open(bb_path, 'r') as f:
        bb_config = yaml.full_load(f)

    update_recursive(cfg, bb_config)

def load_scene_list(scenes_list_file):
    with open(scenes_list_file,'r') as sf:
        scenes = sf.readlines()
    scenes = [s.strip() for s in scenes]

    return scenes

def load_invalid_frames(invalid_frames_file):
    with open(invalid_frames_file, 'r') as iff:
        invalid_frames = yaml.full_load(iff)
    return invalid_frames

