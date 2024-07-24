import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))

print("import isaac first")
from isaac_validator import IsaacValidator

import pickle
from datetime import datetime

import numpy as np
import torch
import transforms3d
from loguru import logger

from utils.hand_model import HandModel
from utils.metrics import cal_pen, cal_q1
from utils.object_model import ObjectModel
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.utils import try_to_device, try_to_torch, try_to_cpu, try_to_numpy


NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
MESH_DIR = 'data/meshdata'

def sim_test(sim: IsaacValidator, grasp_code: str, hand_param: torch.Tensor, object_sc: torch.Tensor):
    sim.reset_simulator()
    B = hand_param.shape[0]
    # For each test, if the batch size is too large, there might be some error.
    assert B <= 500
    sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                  os.path.join(MESH_DIR, grasp_code, "coacd"), "coacd.urdf")
    hand_param = hand_param.cpu()
    rotations = try_to_numpy(robust_compute_rotation_matrix_from_ortho6d(hand_param[:, 3:9]))
    translations = try_to_numpy(hand_param[:, :3])
    hand_poses = try_to_numpy(hand_param[:, 9:])
    scales = try_to_numpy(object_sc)
    for i in range(B):
        rot = transforms3d.quaternions.mat2quat(rotations[i])
        sim.add_env(rot, translations[i], hand_poses[i], scales[i])
    result = sim.run_sim()
    result = np.array(result).reshape(B, 6)
    return result

def obj2hand_sim(cfg):
    # There is a magic number in setting the gpu 
    # for simulation and the number might change 
    # due to the system setting. I couldn't find 
    # a pattern so far. Therefore, I just hard 
    # code it here and change it manually. The 
    # device for loading the results are set to 
    # be on the same device. If error occurs, you 
    # can also consider putting them onto two devices.
    DEVICE_ID = 6
    
    test_cfg = cfg.MODEL.test
    output_path = cfg.OUTPUT_PATH
    st = cfg.start
    step = cfg.step

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=2048,
        device=device
    )
    print(cfg.DATASET.lower())
    object_model = ObjectModel(data_dir=cfg.DATA.DATA_DIR,
                               mesh_dir=cfg.DATA.MESH_DIR, 
                               pc_num_points=2048, 
                               device=device,
                               dataset_name=cfg.DATASET.lower())

    test_file = None
    if len(cfg.TEST_FILE):
        test_file = cfg.TEST_FILE
    else:
        test_obj2hand_path = pjoin(output_path, "test_obj2hand")
        if os.path.exists(test_obj2hand_path):
            files = os.listdir(test_obj2hand_path)
            files = [f for f in files if "test_obj2hand" in f]
            files = sorted(files, key = lambda x: os.path.getmtime(os.path.join(test_obj2hand_path, x)))
            test_file = files[-1]
            logger.info("Automatically detect test file {}", test_file)
            test_file = pjoin(test_obj2hand_path, test_file)
    
    if test_file is None:
        assert False, "Please provide a test file"
    
    with open(test_file, "rb") as f:
        obj2hand_dict = pickle.load(f)
    key_list = list(obj2hand_dict.keys())
    key_list.sort()
    ed = min(st+step, len(key_list))
    
    ori_file_name = os.path.basename(test_file)
    ori_file_name = os.path.splitext(ori_file_name)[0]
    save_file_name = f"sim_st_{st}_ed_{ed-1}_on_{ori_file_name}_t_{NOW_TIME}.pk"
    
    result_dict = dict()

    success, total = 0, 0
    success_opt, total_opt = 0, 0
    # using the magic device.
    sim = IsaacValidator(gpu=DEVICE_ID)
    i = st
    for grasp_code in key_list[st: ed]:
        hand_dict = obj2hand_dict[grasp_code]
        print(f"{i}: Object {grasp_code}:")
        code = grasp_code

        object_sc_np = hand_dict['object_sc']
        object_sc = try_to_torch(object_sc_np)
        object_model.set_parameters(code + "#-1", object_sc=object_sc)
        
        hand_param_out = hand_dict['hand_param_out']
        hand_param_opt = hand_dict['hand_param_opt']
        hand_param_out = try_to_device(try_to_torch(hand_param_out), device)
        hand_param_opt = try_to_device(try_to_torch(hand_param_opt), device)
        hB = hand_param_out.shape[0]
        
        # To save time, we test them together.
        hand_for_test = torch.cat([hand_param_out, hand_param_opt], dim=0)

        object_model.extend()
        hand_model.set_parameters(hand_for_test)
        pc_for_test = object_model.object_pc
        sc_for_test = try_to_cpu(object_model.object_sc)
        B = hand_for_test.shape[0]
        
        # calc e_pen
        E_pen = cal_pen(hand_model, pc_for_test)
        E_pen = E_pen.detach().cpu().numpy()
        penetration = (E_pen < test_cfg.thres_pen)
        
        # calc q1
        q1, contact_points, contact_normals = cal_q1(test_cfg, B, object_model, hand_model, device, with_contact=True)
        q1 = q1.detach().cpu().numpy()
        
        # simulate
        sim_result = sim_test(sim, code, hand_for_test, sc_for_test)
        sim_result = np.sum(sim_result, axis=-1)
        simulated = (sim_result > 0)
        valid = penetration * simulated
        
        penetration = np.split(penetration, [hB])
        sim_result = np.split(sim_result, [hB])
        simulated = np.split(simulated, [hB])
        valid = np.split(valid, [hB])
        q1 = np.split(q1, [hB])
        E_pen = np.split(E_pen, [hB])
        
        result = {
            'object_sc': object_sc_np,
            'hand_param_out': hand_dict['hand_param_out'],
            'hand_param_opt': hand_dict['hand_param_opt'],
            
            'E_pen': E_pen[0],
            'penetration': penetration[0],
            'simulated': sim_result[0],
            'valid': valid[0],
            'q1': q1[0],
            
            'E_pen_opt': E_pen[1],
            'penetration_opt': penetration[1],
            'simulated_opt': sim_result[1],
            'valid_opt': valid[1],
            'q1_opt': q1[1],
            
            'disc_pred': hand_dict['disc_pred'],
        }
        result_dict[grasp_code] = result
        
        # print
        print("Before:")
        print(f"  penetration suc rate: {penetration[0].mean()}")
        print(f"  simulated suc rate: {simulated[0].mean()}")
        print(f"  valid rate: {valid[0].mean()}")
        print(f"  avg q1: {q1[0][penetration[0]].mean()}")
        print(f"  avg pen: {E_pen[0].mean()}")
        print("After Optimization:")
        print(f"  penetration suc rate: {penetration[1].mean()}")
        print(f"  simulated suc rate: {simulated[1].mean()}")
        print(f"  valid rate: {valid[1].mean()}")
        print(f"  avg q1: {q1[1][penetration[1]].mean()}")
        print(f"  avg pen: {E_pen[1].mean()}")

        success += valid[0].sum()
        total += hB

        success_opt += valid[1].sum()
        total_opt += (B - hB)
        
        i += 1

    print(f"Total for current batch, avg success {success / total}")
    print(f"Total for current batch, avg opt success {success_opt / total_opt}")
    if cfg.folder is not None:
        os.makedirs(pjoin(output_path, "test_obj2hand", cfg.folder), exist_ok=True)
        save_path = pjoin(output_path, "test_obj2hand", cfg.folder, save_file_name)
    else:
        save_path = pjoin(output_path, "test_obj2hand", save_file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done test.")
    


if __name__ == '__main__':
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    args = parse_args('Eval in Simulation')

    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    
    # Our experiments shows that the simulation will be broken 
    # after 30 times of reset. Therefore, we use a script to 
    # restart it every 25 tests. 

    file_suffix = f"_st_{args.start}_ed_{args.start + args.step - 1}"
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    file_suffix += f"_{NOW_TIME}"
    full_log_name = f"sim_log{file_suffix}"
    
    cfg.start = args.start
    cfg.step = args.step
    cfg.folder = args.folder

    if args.folder is not None:
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, "sim_log", args.folder), exist_ok=True)
        log_path = os.path.join(cfg.OUTPUT_PATH, "sim_log", args.folder, f"{full_log_name}.log")
    else:
        log_path = os.path.join(cfg.OUPTUT_PATH, "sim_log", f"{full_log_name}.log")

    with DupStdoutFileManager(log_path) as _:
        obj2hand_sim(cfg)

