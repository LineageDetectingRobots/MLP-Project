import os
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
from imageio import imread
from skimage.transform import resize
import argparse
import re

from tqdm import tqdm
import torch
import torch.nn as nn

import framework.networks.arcface.my_face_verify as arcface
from framework.networks.sphereface.my_sphereface import get_model as get_sphereface_model
from framework.networks.inception_resnet_v1 import get_vgg_face2_model
from framework.networks.vgg_face import VGG16
from framework.utils.downloads import download_vgg_weights
from framework import MODEL_PATH
from framework import DATASET_PATH, RESULTS_PATH


def get_model(model_name: str) -> nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'arc_face':
        threshold = 1.54
        model = arcface.get_model(threshold)
    elif model_name == 'vgg_face':
        download_vgg_weights()
        model = VGG16()
        model.to(device)
        # Load model weights that have been just downloaded
        model_path = os.path.join(MODEL_PATH, "vgg_face_torch", "VGG_FACE.t7")
        model.load_weights(model_path)
        # Set model to eval mode when testing/evaluating
        model.eval()
    elif model_name == 'sphere_face':
        model = get_sphereface_model()
        model.to(device)
        model.eval()
    elif model_name == 'vgg_face2':
        model = get_vgg_face2_model()
        model.to(device)
    else:
        raise RuntimeError(f'unknown model name {model_name}')

    return model

def get_unique_images(dataframe: pd.DataFrame):
    return (dataframe.F.append(dataframe.M).append(dataframe.C)).unique()

def l2_normalisation(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def normalise(imgs):
    # TODO: const values for whole set not just batch
    if imgs.ndim == 4:
        axis = (1, 2, 3)
        size = imgs[0].size
    elif imgs.ndim == 3:
        axis = (0, 1, 2)
        size = imgs.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(imgs, axis=axis, keepdims=True)
    std = np.std(imgs, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    normalised_imgs = (imgs - mean) / std_adj
    return normalised_imgs

def calc_features(model, photo_paths: list, batch_size=128):
    device = model._device()

    pred_features = []
    all_paths = []
    for start in tqdm(range(0, len(photo_paths), batch_size)):
        batch_paths = photo_paths[start:start + batch_size]
        image_batch, paths = model.load_and_resize_images(batch_paths)
        all_paths = all_paths + paths

        # TODO: Normalise image batch? Add model
        image_batch = normalise(image_batch)

        # Move image batch to device
        image_inputs = torch.from_numpy(image_batch).float().to(device)
        feature_batch = model.get_features(image_inputs)
        feature_batch_numpy = feature_batch.cpu().numpy()
        pred_features.append(feature_batch_numpy)
    # TODO: Do l2 normalisation???
    features = l2_normalisation(np.concatenate(pred_features))
    return features, all_paths

def get_base_filepath(paths):
    base_filepaths = []
    for path in paths:
        result = re.search('FIDs/(.*)', path).group(1)
        base_filepaths.append(result)
    return base_filepaths

def remove_rows(cross_val_df, paths, all_paths):
    base_filepaths = get_base_filepath(paths)
    removed_paths = get_base_filepath(all_paths)
    no_img_path = []
    for path in removed_paths:
        if path not in base_filepaths:
            no_img_path.append(path)

    remove_idxs = set()
    for idx, row in tqdm(cross_val_df.iterrows()):
        for path in no_img_path:
            if row.F == path or row.M == path or row.C == path:
                remove_idxs.add(idx)
                print('removing row = {}'.format(idx))
                break
            
    print('num_rows_removed = ,', len(list(remove_idxs)))
    new_df = cross_val_df.drop(list(remove_idxs))
    return new_df
            

def get_filepath_to_vector(feature_vecs, filepaths):
    base_filepaths = get_base_filepath(filepaths)
    conversion_dict = {}
    for i, base_path in enumerate(base_filepaths):
        conversion_dict[base_path] = feature_vecs[i]
    return conversion_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-m", "--model", help="which face rec model", default='vgg_face2', type=str)
    parser.add_argument("-o", "--overwrite", help="Overwrite feature vec mapping", dest='overwrite', action='store_true')
    # TODO: test vggface2, vgg_face

    args = parser.parse_args()
    overwrite = args.overwrite

    model_name = args.model
    model = get_model(model_name)

    # Get our file we are using for testing
    cross_val_csv = os.path.join(DATASET_PATH, 'fiw', 'tripairs', '5_cross_val.csv')
    new_val_csv = os.path.join(DATASET_PATH, 'fiw', 'tripairs', f'{model_name}_5_cross_val.csv')
    cross_val_df = pd.read_csv(cross_val_csv)
    # print(cross_val_df)
    unique_photo_filepaths = get_unique_images(cross_val_df)
    # print(unique_photo_filepaths)
    
    with torch.no_grad():
        feature_vec_results = os.path.join(RESULTS_PATH, f'mappings_{model_name}.pickle')
        if (not os.path.exists(feature_vec_results)) or overwrite:
            photo_folder = os.path.join(DATASET_PATH, 'fiw', 'FIDs')
            # print(photo_folder)
            all_paths = [os.path.join(photo_folder, photo) for photo in unique_photo_filepaths]
            feature_vecs, paths = calc_features(model, all_paths)
            # Some photos might not be preprocessed, we need to update the csv file to reflect this
            new_csv = remove_rows(cross_val_df, paths, all_paths)
            new_csv.to_csv(new_val_csv)
            # Create a mapping from img_filepath to vector
            filepath_map = get_filepath_to_vector(feature_vecs, paths)
            # save all as one tuple
            data = (feature_vecs, filepath_map)
            with open(feature_vec_results, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise RuntimeError(f'Feature vectors already exist for model {model_name}')
