import os, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
from skimage.transform import resize
from imageio import imread
from tqdm import tqdm
from framework import MODEL_PATH


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
        # https://www.kaggle.com/ateplyuk/vggface-baseline-in-keras?fbclid=IwAR12q3OsejMsQFggEWl1Rb2LcgIeKBszW8k6fQ3pEFPv5pvF-t3dBLtKPY0
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
        aligned = resize(img, (image_size, image_size))
        resized_images.append(aligned)
    return resized_images

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


def calc_features(model: VGG16, filepaths, batch_size=128):
    pred_features = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        normalised_images = normalise(load_and_resize_images(filepaths[start:start+batch_size]))
        pred_features.append(model.get_features(normalised_images))
    # TODO: l2 norm
    features = l2_normalisation(np.concatenate(pred_features))
    return features

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

if __name__ == "__main__":
    download_vgg_weights()
    model = VGG16()
    model_path = os.path.join(MODEL_PATH, "vgg_face_torch", "VGG_FACE.t7")
    model.load_weights(model_path)
