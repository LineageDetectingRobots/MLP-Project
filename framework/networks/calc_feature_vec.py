import os
import pandas as pd
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import arcface.my_face_verify as arcface
# import as sphereface
# import as facenet
from vgg_face import VGG16
from framework.utils.downloads import download_vgg_weights
from framework import MODEL_PATH


from framework import DATASET_PATH, RESULTS_PATH

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

    return model

def get_unique_images(dataframe: pd.DataFrame):
    return dataframe.F.append(dataframe.M).append(dataframe.C).unique()

def l2_normalisation(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_resize_images(filepaths, image_size=244):
    resized_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')

        resized_images.append(aligned)
    return np.array(resized_images).reshape(-1, 3, image_size, image_size)

def normalise(imgs):
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

def calc_features(model, photo_paths: list, batch_size=64):
    # TODO: Find out what device the model is on
    device = model._device()
    pred_features = []
    for start in tqdm(range(0, len(photo_paths), batch_size)):
        # TODO: load and preprocess images
        batch_paths = photo_paths[start:start + batch_size]
        # TODO: Make image size configurable
        image_batch = load_and_resize_images(batch_paths, image_size = 255)

        # TODO: Normalise image batch?
        image_batch = normalise(image_batch)

        # Move image batch to device
        image_inputs = torch.from_numpy(image_batch).float().to(device)
        # TODO: call correctly, could be forward
        feature_batch = model.get_features(image_inputs)
        feature_batch_numpy = feature_batch.cpu().numpy()
        pred_features.append(feature_batch_numpy)
    # TODO: Do l2 normalisation???
    features = l2_normalisation(np.concatenate(pred_features))
    return features





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-m", "--model", help="which face rec model", default='arc_face', type=str)
    args = parser.parse_args()

    model_name = args.model
    model = get_model(model_name)

    # Get our file we are using for testing
    cross_val_csv = os.path.join(DATASET_PATH, 'fiw', 'tripairs', '5_cross_val.csv')
    cross_val_df = pd.read_csv(cross_val_csv)
    unique_photo_filepaths = get_unique_images(cross_val_df)
    print(unique_photo_filepaths)
    # return
    
    with torch.no_grad():
        feature_vec_results = os.path.join(RESULTS_PATH, f'feature_vec_{model_name}.npy')
        if not os.path.exists(feature_vec_results):
            photo_folder = os.path.join(DATASET_PATH, 'fiw', 'FIDs')
            feature_vecs = calc_features(model, [os.path.join(photo_folder, photo) for photo in unique_photo_filepaths])
            np.save(feature_vec_results, feature_vecs)
        else:
            raise RuntimeError(f'Feature vectors already exist for model {model_name}')
