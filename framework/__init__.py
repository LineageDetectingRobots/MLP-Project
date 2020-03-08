
import os 
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO: Check if on MLP or not
parser = argparse.ArgumentParser()
parser.add_argument('--cluster', metavar='C', type=int, nargs='+',
                    help='1 for mlp cluster, otherwise 0', default=0, dest='cluster')
args = parser.parse_args()

if args.cluster == 1:
    DATASET_PATH = os.getenv('DATASET_DIR')
    if DATASET_PATH is None:
        raise RuntimeError("Dataset path is None on MLP cluster")
else:
    DATASET_PATH = os.path.join(dir_path, "..", "datasets")
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

if args.cluster == 1:
    MODEL_PATH = os.getenv('MODEL_DIR')
    if MODEL_PATH is None:
        raise RuntimeError("Model path is None on MLP cluster")
else:
    MODEL_PATH = os.path.join(dir_path, "..", "models")
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

if args.cluster == 1:
    RESULTS_PATH = os.getenv('RESULTS_DIR')
    if RESULTS_PATH is None:
        raise RuntimeError("Results path is None on MLP cluster")
else:
    RESULTS_PATH = os.path.join(dir_path, "..", "results")
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)


ARC_FACE = os.path.join(dir_path, "networks", "arcface")
if not os.path.exists(ARC_FACE):
    os.mkdir(ARC_FACE)


del os