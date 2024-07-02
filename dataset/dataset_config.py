from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

__C.DEXGRASPNET = edict()
__C.DEXGRASPNET.DATA_DIR = 'data/dexgraspnet'
__C.DEXGRASPNET.MESH_DIR = 'data/meshdata'
__C.DEXGRASPNET.SPLITS_DIR = 'data/splits'
__C.DEXGRASPNET.PC_NUM_POINTS = 2048
__C.DEXGRASPNET.ROT_TYPE = 'mat'  # [euler, quat, mat]

__C.DEXGRASPNET.TRAIN_LENGTH = -1
__C.DEXGRASPNET.TEST_LENGTH = -1
__C.DEXGRASPNET.HAND_RATIO = -1
__C.DEXGRASPNET.OBJECT_RATIO = -1

__C.DEXGRASPNET.USE_PRECOMPUTE_PC = False
__C.DEXGRASPNET.NORMALIZE_PC = False
__C.DEXGRASPNET.USE_POINT_CLOUD = False
__C.DEXGRASPNET.APPLY_RANDOM_ROT = False