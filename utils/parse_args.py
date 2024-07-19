import argparse
import os
import platform

from .config import cfg, cfg_from_file, cfg_from_list, generate_output_path
from .utils import cp_some


def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    # test only
    parser.add_argument('--obj2hand_test', action='store_true', default=False,
                        help='iterate over object set and save o2h result')
    parser.add_argument('--hand2obj_test', action='store_true', default=False,
                        help='iterate over hand set and save h2o result')
    parser.add_argument('--joint_test', action='store_true', default=False,
                        help='iterate over hand set and save h2o result')
    parser.add_argument('--obj2hand_sim', action='store_true', default=False,
                        help='use simulation test hand')
    parser.add_argument('--start', default=0, type=int,
                        help='start iterations')
    parser.add_argument('--folder', default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    if len(cfg.MODEL_NAME) != 0:
        output_path, model_save_path = generate_output_path(cfg.MODEL_NAME, cfg.PROJECT)
        cfg_from_list(['OUTPUT_PATH', output_path, 'MODEL_SAVE_PATH', model_save_path])
        if not os.path.exists(cfg.OUTPUT_PATH):
            os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
        if not os.path.exists(cfg.MODEL_SAVE_PATH):
            os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)

    if args.cfg_file is not None:
        # Save the config file into the model save path
        for f in args.cfg_file:
            cp_some(f, cfg.OUTPUT_PATH)

    return args
