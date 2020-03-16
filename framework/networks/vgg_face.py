import os, sys, argparse
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
from framework.utils.downloads import download_vgg_weights
from PIL import Image
from torchvision import transforms as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    
    def _device(self):
        return next(self.parameters()).device

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
                
    def load_and_resize_images(self, batch_paths, transform):
        image_size = (244, 244)
        transform = transform(image_size)
        resized_imgs = []
        with torch.no_grad():
            for path in batch_paths:
                img = Image.open(path)
                resized_img = transform(img)
                resized_imgs.append(resized_img)
        stacked = torch.stack(resized_imgs)
        return stacked

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


def l2_normalisation(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calc_features(model: VGG16, filepaths, batch_size=128):
    device = model._device()

    # NOTE: transforms from the FIW GitHub:
    # https://github.com/visionjo/FIW_KRT/blob/master/sphereface_rfiw_baseline/data_loader.py
    transform = lambda image_size : T.Compose([T.RandomHorizontalFlip(),
                                    T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])

    pred_features = []
    device = model._device()
    for start in tqdm(range(0, len(filepaths), batch_size)):
        normalised_images = model.load_and_resize_images(filepaths[start:start+batch_size], transform)
        # Need to convert to a tensor
        inputs = normalised_images.float().to(device)
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
    folder_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs')
    val_csv = os.path.join(folder_path, "5_cross_val.csv")
    val_df = pd.read_csv(val_csv)
    unique_photos_filepaths = get_unique_images(val_df)

    datapath = os.path.join(DATASET_PATH, 'fiw')
    with torch.no_grad():
        test_vgg_results = os.path.join(RESULTS_PATH, "test_fiw_vgg.npy")
        if not os.path.exists(test_vgg_results):
            test_features = calc_features(model, [os.path.join(datapath, "FIDs", file) for file in unique_photos_filepaths])
            np.save(test_vgg_results, test_features)
        else:
            test_features = np.load(test_vgg_results)
    
    # Create a mapping for image filepath to index
    img2idx = dict()
    for idx, img_filepath in enumerate(unique_photos_filepaths):
        img2idx[img_filepath] = idx
    
    # Create a distance column in test_df. Euclidean distance between image 
    val_df["distance"] = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        imgs_F = [test_features[img2idx[row.F]], test_features[img2idx[row.C]]]
        imgs_M = [test_features[img2idx[row.M]], test_features[img2idx[row.C]]]
        val_df.loc[idx, "distance_F"] = distance.euclidean(*imgs_F)
        val_df.loc[idx, "distance_M"] = distance.euclidean(*imgs_M)

    # Sum all the distances up
    distances_F = val_df.distance_F.values
    distances_M = val_df.distance_M.values
    sum_dist_F = np.sum(distances_F)
    sum_dist_M = np.sum(distances_M)

    # Calculate prob. based on sum of all closer matches over total distance summed
    probs = []
    for (dist_F, dist_M) in tqdm(zip(distances_F, distances_M)):
        prob_F = np.sum(distances_F[np.where(distances_F <= dist_F)[0]])/sum_dist_F
        prob_M = np.sum(distances_M[np.where(distances_M <= dist_M)[0]])/sum_dist_M
        prob = (prob_F + prob_M) / 2.0
        probs.append(1 - prob)
    
    val_df["probs"] = probs

    val_df["pred_label"] = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        if row.probs > 0.5:
            val_df.loc[idx, "pred_label"] = 1
        else:
            val_df.loc[idx, "pred_label"] = 0
    
    save_results_path = os.path.join(RESULTS_PATH, "results_baseline.csv")
    val_df.to_csv(save_results_path)
    
def get_unique_images(dataframe):
    return (dataframe.F.append(dataframe.M).append(dataframe.C)).unique()



if __name__ == "__main__":
    # Example: run one of the below functions to do things and stuff
    # run_sample_submission()
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('cluster', metavar='M', type=int, nargs='+',
    #                 help='1 for mlp cluster, otherwise 0', default=0)
    run_train_baseline()
