from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_SPLIT_DIR = BASE_DIR / "data_split"
TRAIN_DIR = DATA_SPLIT_DIR / "train"
VAL_DIR = DATA_SPLIT_DIR / "val"
TEST_DIR = DATA_SPLIT_DIR / "test"

A2_MODELS_DIR = BASE_DIR / "a2_models"
A2_REPORTS_DIR = BASE_DIR / "a2_reports"
A2_FIGURES_DIR = A2_REPORTS_DIR / "figures"

A2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
A2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
A2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOTA_MODEL_PATH = A2_MODELS_DIR / "convnext_sota.pth"
ABLATION_MODEL_PATH = A2_MODELS_DIR / "convnext_no_artaug.pth"
CLASSES_MAP_PATH = BASE_DIR / "models" / "classes.json"

CLASSES = ["paper", "rock", "scissors"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
