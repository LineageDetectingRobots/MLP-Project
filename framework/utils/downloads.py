import os
import zipfile
import kaggle

from framework import DATASET_PATH

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
            os.system(f'mv {vgg_folder} {MODEL_PATH}')
        except:
            raise
        finally:
            print("Cleaning up files...")
            os.system(f'rm {tar_filepath}') 
            os.system(f'rm -rf {vgg_folder}')


def download_fiw_kaggle():
    try:
        kaggle.api.authenticate()
    except:
        raise RuntimeError('The API key can be found on kaggle in account. Add the following environment variables. \n' + \
                           'e.g. export KAGGLE_USERNAME=username, \n' + \
                          'export export KAGGLE_KEY=xxxxxxxxxxxxx')
    cur_dir = os.getcwd()
    fiw_kaggle_zip = os.path.join(cur_dir, "recognizing-faces-in-the-wild.zip")
    fiw_folder = "recognizing-faces-in-the-wild"
    dataset_fiw = os.path.join(DATASET_PATH, fiw_folder)
    if not os.path.exists(dataset_fiw):
        try:
            # Create recognizing-faces-in-the-wild in datasets
            if not os.path.exists(dataset_fiw):
                os.mkdir(dataset_fiw)
            
            # Download the dataset files
            os.system('kaggle competitions download -c recognizing-faces-in-the-wild')

            print("Extracting files....")
            # Unzip and move to datasets/recognizing-faces-in-the-wild
            with zipfile.ZipFile(fiw_kaggle_zip, 'r') as zip_f:
                zip_f.extractall(dataset_fiw)
            
            # Extract all remaining zip files
            for file in os.listdir(dataset_fiw):
                if file.endswith(".zip"):
                    # Create new folder, extract into new folder, delete old zip
                    zip_filepath = os.path.join(dataset_fiw, file)
                    new_folder = os.path.join(dataset_fiw, file[:-4])
                    os.mkdir(new_folder)
                    with zipfile.ZipFile(zip_filepath, 'r') as zip_f:
                        zip_f.extractall(new_folder)
                    os.remove(zip_filepath)
        except:
            os.rmdir(dataset_fiw)
        finally:
            os.remove(fiw_kaggle_zip)

if __name__ == "__main__":
    # Example please delete if annoying
    download_fiw_kaggle()