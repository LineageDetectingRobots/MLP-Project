import os
import zipfile
from framework import DATASET_PATH, MODEL_PATH

def download_vgg_weights():
    # Downloads the vgg model and extracts it
    curr_path = os.getcwd()
    tar_filepath = os.path.join(curr_path, 'vgg_face_torch.tar.gz')
    vgg_folder = os.path.join(curr_path, "vgg_face_torch")

    model_vgg_folder = os.path.join(MODEL_PATH, "vgg_face_torch")
    if not os.path.exists(model_vgg_folder):
        try:
            print("Downloading....")
            os.system('wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz')
            print("Extracting....")
            os.system('tar -xvf vgg_face_torch.tar.gz')
            os.system(f'mv {vgg_folder} {model_vgg_folder}')
        except:
            os.rmdir(model_vgg_folder)
            raise
        finally:
            print("Cleaning up files...")
            os.system(f'rm {tar_filepath}') 
            os.system(f'rm -rf {vgg_folder}')
