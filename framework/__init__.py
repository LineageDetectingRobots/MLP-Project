
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

DATASET_PATH = os.path.join(dir_path, "..", "datasets")
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

MODEL_PATH = os.path.join(dir_path, "..", "models")
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

del os