from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_SPLIT_DIR = BASE_DIR / "data_split"
TRAIN_DIR = DATA_SPLIT_DIR / "train"
VAL_DIR = DATA_SPLIT_DIR / "val"
TEST_DIR = DATA_SPLIT_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"


MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


SVM_MODEL_PATH = MODELS_DIR / "svm_baseline.joblib"
CNN_MODEL_PATH = MODELS_DIR / "custom_cnn.keras"
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenetv2_finetuned.keras"
CLASSES_MAP_PATH = MODELS_DIR / "classes.json"


CLASSES = ["paper", "rock", "scissors"]
NUM_CLASSES = len(CLASSES)


IMG_SIZE = (224, 224)
IMG_SIZE_SVM = (64, 64)
BATCH_SIZE = 32
RANDOM_SEED = 42


EPOCHS_CNN = 20
EPOCHS_MOBILENET_P1 = 12
EPOCHS_MOBILENET_P2 = 15
LR_INITIAL = 1e-3
LR_FINETUNE = 1e-5
