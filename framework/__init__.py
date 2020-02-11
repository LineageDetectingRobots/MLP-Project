
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

DATASET_PATH = os.path.join(dir_path, "..", "datasets")
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

MODEL_PATH = os.path.join(dir_path, "..", "models")
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

RESULTS_PATH = os.path.join(dir_path, "..", "results")
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

del os