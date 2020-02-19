import os
from framework import DATASET_PATH

def gen_val_set():
    fiw_path = os.path.join(DATASET_PATH, "fiw")
    if not os.path.exists(fiw_path):
        
