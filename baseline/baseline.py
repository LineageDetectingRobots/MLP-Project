import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from tensorflow.keras.models import load_model
# from keras.models import load_model
import pandas as pd
from tqdm import tqdm

from framework import DATASET_PATH

# NOTE: Dataset file path should be datasets/"dataset_name"/<csvs or image data>
train_df = pd.read_csv(os.path.join(DATASET_PATH,
                                    "recognizing-faces-in-the-wild",
                                    "train_relationships.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH,
                                    "recognizing-faces-in-the-wild",
                                    "sample_submission.csv"))

from keras_facenet import FaceNet
# model = FaceNet()

current_directory = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_directory, 'facenet_keras.h5')
model = load_model(model_path)

# Here's preprocessing stuff. The images from the test set seem 
# to already be aligned, so I'll omit that part here for clarity.

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin,image_size = 160):
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)

# Here we'll compute all the embeddings 
# for the test images using the pretrained model

# https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-images.ipynb
# Can someone help me fix why this predict on batch does not work, its taken from the docs

def calc_embs(filepaths, margin=10, batch_size=512):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size], margin))
        pd.append(model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs

test_filepath = os.path.join(DATASET_PATH,
                             "recognizing-faces-in-the-wild",
                             "test")
test_images = os.listdir(test_filepath)
test_embs = calc_embs([os.path.join(test_filepath, f) for f in test_images])
np.save("test_embs.npy", test_embs)


test_df["distance"] = 0
img2idx = dict()
for idx, img in enumerate(test_images):
    img2idx[img] = idx

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    imgs = [test_embs[img2idx[img]] for img in row.img_pair.split("-")]
    test_df.loc[idx, "distance"] = distance.euclidean(*imgs)

all_distances = test_df.distance.values
sum_dist = np.sum(all_distances)

probs = []
for dist in tqdm(all_distances):
    prob = np.sum(all_distances[np.where(all_distances <= dist)[0]])/sum_dist
    probs.append(1 - prob)

sample_filepath = os.path.join(DATASET_PATH,
                               "recognizing-faces-in-the-wild",
                               "sample_submission.csv")
sub_df = pd.read_csv(sample_filepath)
sub_df.is_related = probs
sub_df.to_csv("submission.csv", index=False)