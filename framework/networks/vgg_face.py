import os, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import pandas as pd
from skimage.transform import resize
from scipy.spatial import distance
from imageio import imread
from tqdm import tqdm
from framework import MODEL_PATH, DATASET_PATH, RESULTS_PATH
from framework.utils.downloads import download_vgg_weights, download_fiw_kaggle

# TODO: Enable this if you have a decent GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class VGG16(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
    
    def load_weights(self, path):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
    
    def forward(self, x):
        """
        input image (244x244)
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)
    
    def get_features(self, x):
        """
        When top layer is removed can be used to get similarity metric between faces
        """
        # TODO: Other study in Kaggle comp. used avg pool
        # https://www.kaggle.test_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")om/ateplyuk/vggface-baseline-in-keras?fbclid=IwAR12q3OsejMsQFggEWl1Rb2LcgIeKBszW8k6fQ3pEFPv5pvF-t3dBLtKPY0
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return F.relu(self.fc7(x))
    
def load_and_resize_images(filepaths, image_size=244):
    resized_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')

        resized_images.append(aligned)
    # print(len(resized_images))
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


def l2_normalisation(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calc_features(model: VGG16, filepaths, batch_size=64):
    pred_features = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        normalised_images = normalise(load_and_resize_images(filepaths[start:start+batch_size]))
        # Need to convert to a tensor
        inputs = torch.from_numpy(normalised_images).float().to(device)
        features = model.get_features(inputs)
        features_numpy = features.cpu().numpy()
        pred_features.append(features_numpy)
    features = l2_normalisation(np.concatenate(pred_features))
    return features



def run_sample_submission():
    download_fiw_kaggle()
    download_vgg_weights()
    model = VGG16()
    model.to(device)
    # Load model weights that have been just downloaded
    model_path = os.path.join(MODEL_PATH, "vgg_face_torch", "VGG_FACE.t7")
    model.load_weights(model_path)

    # Set model to eval mode when testing/evaluating
    model.eval()

    # Get test df
    submission_csv = os.path.join(DATASET_PATH, "recognizing-faces-in-the-wild", "sample_submission.csv")
    test_df = pd.read_csv(submission_csv)

    # Get test images filpath
    test_filepath = os.path.join(DATASET_PATH, "recognizing-faces-in-the-wild", "test")
    test_images = os.listdir(test_filepath)
    # Calculate the features and save them, this takes a long time so best to save, when you have done it
    with torch.no_grad():
        test_vgg_results = os.path.join(RESULTS_PATH, "test_embs_vgg.npy")
        if not os.path.exists(test_vgg_results):
            test_features = calc_features(model, [os.path.join(test_filepath, file) for file in test_images])
            np.save(test_vgg_results, test_features)
        else:
            test_features = np.load(test_vgg_results)
    
    # Create a mapping for image filepath to index
    img2idx = dict()
    for idx, img_filepath in enumerate(test_images):
        img2idx[img_filepath] = idx
    
    # Create a distance column in test_df. Euclidean distance between image pairs
    test_df["distance"] = 0
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        imgs = [test_features[img2idx[img]] for img in row.img_pair.split("-")]
        test_df.loc[idx, "distance"] = distance.euclidean(*imgs)

    # Sum all the distances up
    all_distances = test_df.distance.values
    sum_dist = np.sum(all_distances)

    # Calculate prob. based on sum of all closer matches over total distance summed
    probs = []
    for dist in tqdm(all_distances):
        prob = np.sum(all_distances[np.where(all_distances <= dist)[0]])/sum_dist
        probs.append(1 - prob)

def run_train_baseline():
    download_vgg_weights()
    model = VGG16()
    model.to(device)
    # Load model weights that have been just downloaded
    model_path = os.path.join(MODEL_PATH, "vgg_face_torch", "VGG_FACE.t7")
    model.load_weights(model_path)

    # Set model to eval mode when testing/evaluating
    model.eval()

    # Get test df
    folder_path = os.path.join(DATASET_PATH, 'fiw')
    val_csv = os.path.join(folder_path, "new_val.csv")
    val_df = pd.read_csv(val_csv)
    unique_photos_filepaths = get_unique_images(val_df)

    with torch.no_grad():
        test_vgg_results = os.path.join(RESULTS_PATH, "test_fiw_vgg.npy")
        if not os.path.exists(test_vgg_results):
            test_features = calc_features(model, [os.path.join(folder_path, "FIDs", file) for file in unique_photos_filepaths])
            np.save(test_vgg_results, test_features)
        else:
            test_features = np.load(test_vgg_results)
    
    # Create a mapping for image filepath to index
    img2idx = dict()
    for idx, img_filepath in enumerate(unique_photos_filepaths):
        img2idx[img_filepath] = idx
    
    # Create a distance column in test_df. Euclidean distance between image pairs
    val_df["distance"] = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        imgs = [test_features[img2idx[img]] for img in row.img_pair.split("-")]
        val_df.loc[idx, "distance"] = distance.euclidean(*imgs)

    # Sum all the distances up
    all_distances = val_df.distance.values
    sum_dist = np.sum(all_distances)

    # Calculate prob. based on sum of all closer matches over total distance summed
    probs = []
    for dist in tqdm(all_distances):
        prob = np.sum(all_distances[np.where(all_distances <= dist)[0]])/sum_dist
        probs.append(1 - prob)
    
    val_df["probs"] = probs
    # val_df.loc[idx, "distance"] = distance.euclidean(*imgs)
    



def get_unique_images(dataframe):
    return dataframe.p1.append(dataframe.p2).unique()



if __name__ == "__main__":
    # Example: run one of the below functions to do things and stuff
    # run_sample_submission()
    run_train_baseline()
