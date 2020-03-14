
import os
import pickle
from pprint import pprint
from skimage import io
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt

import torch

class FIW(data.Dataset):
    def __init__(self, path_to_split_csv: str, path_to_mappings: str,  folds: list):
        # NOTE: Expected that dataset already contains 50/50 split of relation / non-relation
        self.feature_vecs, self.mappings = self.get_mappings_and_feature_vec(path_to_mappings)

        # Expects cross fold
        self.dataset = pd.read_csv(path_to_split_csv)
        drop_idxs = []
        for idx, row in self.dataset.iterrows():
            if row.fold not in folds:
                drop_idxs.append(idx)
        self.dataset.drop(self.dataset.index[drop_idxs], inplace=True)

    def get_mappings_and_feature_vec(self, path_to_mappings: str):
        with open(path_to_mappings, 'rb') as file:
            feature_vecs, mappings = pickle.load(file)
        return feature_vecs, mappings
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_label(self, idx):
        # CHECK IF FROM THE SAME FAMILY
        father_fam_id = self.dataset.iloc[idx].F[:5]
        mother_fam_id = self.dataset.iloc[idx].M[:5]
        child_fam_id = self.dataset.iloc[idx].C[:5]
        if father_fam_id == mother_fam_id:
            if father_fam_id == child_fam_id:
                return 1
            else:
                return 0
        else:
            raise RuntimeError('Mother and father family ID should be the same. index = ', idx)
    
    def _get_tripair(self, idx):
        father_path = self.dataset.iloc[idx].F
        mother_path = self.dataset.iloc[idx].M
        child_path = self.dataset.iloc[idx].C

        father = torch.FloatTensor(self.mappings[father_path])
        mother = torch.FloatTensor(self.mappings[mother_path])
        child = torch.FloatTensor(self.mappings[child_path])
        fam_list = [father, mother, child]
        fam_vec = torch.stack(fam_list).view(-1)
        return fam_vec

    def __getitem__(self, idx):
        return self._get_tripair(idx), self._get_label(idx)

if __name__ == "__main__":
    # EXAMPLE USAGE for tripairs
    from framework import DATASET_PATH, RESULTS_PATH
    # NOTE: Netowrk name is required to known which network produces the feature vectors
    model_name = 'sphere_face'
    csv_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs', f'{model_name}_5_cross_val.csv')
    mappings_path = os.path.join(RESULTS_PATH, f'mappings_{model_name}.pickle')
    train_folds = [1, 2, 3, 4]
    test_folds = [5]
    train_dataset = FIW(csv_path, mappings_path, train_folds) 
    test_dataset = FIW(csv_path, mappings_path, test_folds)
    fam_vec, label = train_dataset.__getitem__(128)
    print('is_fam = ', label)
    print('fam_vec = {}'.format(fam_vec.size()))
