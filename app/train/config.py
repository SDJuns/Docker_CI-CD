import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

params = {
    'LR': 0.1e-3,
    'BATCH_SIZE': 32,
    'EPOCH': 10,
    'TRAIN_RATIO': 0.8,
    'LR_STEP': 3,
    'LR_GAMMA': 0.9
}

MODEL_WEIGHT_PATH = os.path.join(BASE_PATH, "app/model_weight/final_best_model_v2.pth")
DATA_PATH = os.path.join(BASE_PATH, "data")
SAVE_MODEL_PATH = os.path.join(BASE_PATH, "app/model_weight/final_best_model.pth")
SAVE_HISTORY_PATH = os.path.join(BASE_PATH, "app/model_weight/train_history.pth")
