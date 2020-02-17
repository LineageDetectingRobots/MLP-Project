
import os 
# import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO: Check if on MLP or not
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('mlp-cluster', metavar='M', type=int, nargs='+',
                    help='1 for mlp cluster, otherwise 0', default=0)
args = parser.parse_args()
if args.mlp-cluster == 1:
    DATASET_PATH = os.getenv('DATASET_DIR')
    if DATASET_PATH is None:
        raise RuntimeError("Dataset path is None on MLP cluster")
else:
    DATASET_PATH = os.path.join(dir_path, "..", "datasets")
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

if args.mlp-cluster == 1:
    MODEL_PATH = os.getenv('MODEL_DIR')
    if MODEL_PATH is None:
        raise RuntimeError("Model path is None on MLP cluster")
else:
    MODEL_PATH = os.path.join(dir_path, "..", "models")
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

if args.mlp-cluster == 1:
    RESULTS_PATH = os.getenv('RESULTS_DIR')
    if RESULTS_PATH is None:
        raise RuntimeError("Results path is None on MLP cluster")
else:
    RESULTS_PATH = os.path.join(dir_path, "..", "results")
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

del os