import os

SMPL_DATA_PATH = "models/smpl/"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")

SMPLX_MODEL_PATH = os.path.join("/home/ubuntu/libs/smplx/transfer_data/body_models/smplx", "SMPLX_NEUTRAL.pkl")

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
