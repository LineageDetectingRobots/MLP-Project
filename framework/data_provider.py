
import os
from skimage import io
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt

class FIW_kaggle(data.Dataset):
    """

    """
    # TODO: add url when request is accepted
    # TODO: Add auto download when I get detials
    url = ""

    def __init__(self, path_to_dataset: str, setting: str = 'train', transform_data = None):
        """
        Args:
            path_to_dataset: path to the dataset folder
            setting: 'train' or 'test' used to pick the type of dataset
            transform_data: the transformations you wish to use on the raw data
        """

        self.path = path_to_dataset
        self.ds_path = ""
        self.transform = transform_data
        self.dataset = None

        # TODO: Most likely needs to be updated after I get permission to use the dataset
        if setting == "train":
            csv_path = os.path.join(self.path, "train_relationships.csv")
            self.ds_path = os.path.join(self.path, "train")
            
            # Separate dataset into family and individual
            self.dataset = pd.read_csv(csv_path)
            sep_p1 = self.dataset["p1"].str.split("/", n = 1, expand = True)
            self.dataset["family_1"] = sep_p1[0]
            self.dataset["individual_1"] = sep_p1[1]

            sep_p2 = self.dataset["p2"].str.split("/", n = 1, expand = True)
            self.dataset["family_2"] = sep_p2[0]
            self.dataset["individual_2"] = sep_p2[1]

            # Dropping old Name columns
            self.dataset.drop(columns =["p1"], inplace = True)
            self.dataset.drop(columns =["p2"], inplace = True)
        elif setting == 'validation':
            raise RuntimeError("Validation not implemented yet. Sorry")
        elif setting == "test":
            # TODO: Implement
            # csv_path =
            raise RuntimeError("Test not implemented yet. Sorry")
        else:
            raise RuntimeError(f"Setting {setting} unkown for FIW dataset. Please use \'train\' or \'test\'")
        
    
    def _download(self):
        # TODO: Should be used to download dataset if not currently available
        pass
    
    def __len__(self):
        return len(self.dataset)

    def _get_pair(self, idx):
        pair_1 = os.path.join(self.ds_path, self.dataset.iloc[idx, 0], self.dataset.iloc[idx, 1])
        print(self.dataset.iloc[idx])
        pair_2 = os.path.join(self.ds_path, self.dataset.iloc[idx, 2], self.dataset.iloc[idx, 3])
        return (pair_1, pair_2)
    
    def _get_label(self, idx):
        if self.dataset.iloc[idx, 0] == self.dataset.iloc[idx, 2]:
            return 1
        else:
            return 0
    
    def __getitem__(self, idx):

        return idx, self._get_pair(idx), self._get_label(idx)


class FIW(data.Dataset):
    def __init__(self, path_to_dataset: str, path_to_split_csv: str, folds: list):
        # NOTE: Expected that dataset already contains 50/50 split of relation / non-relation
        self.ds_path = path_to_dataset

        # Expects cross fold
        self.dataset = pd.read_csv(path_to_split_csv)
        drop_idxs = []
        for idx, row in self.dataset.iterrows():
            if row.fold not in folds:
                drop_idxs.append(idx)
        self.dataset.drop(self.dataset.index[drop_idxs], inplace=True)

    
    def __len__(self):
        return len(self.dataset)
    
    def _get_label(self, idx):
        # CHECK IF FROM THE SAME FAMILY
        father_fam_id = self.dataset.iloc[idx].F[:5]
        mother_fam_id = self.dataset.iloc[idx].M[:5]
        child_fam_id = self.dataset.iloc[idx].C[:5]
        if father_fam_id == mother_fam_id:
            if father_fam_id == child_fam_id:
                return 1;
            else:
                return 0;
        else:
            raise RuntimeError('Mother and father family ID should be the same. index = ', idx)
    
    def _get_tripair(self, idx):
        father_path = os.path.join(self.ds_path, self.dataset.iloc[idx].F)
        mother_path = os.path.join(self.ds_path, self.dataset.iloc[idx].M)
        child_path = os.path.join(self.ds_path, self.dataset.iloc[idx].C)

        father = io.imread(father_path)
        mother = io.imread(mother_path)
        child = io.imread(child_path)
        return [father, mother, child]

    def __getitem__(self, idx):
        return self._get_tripair(idx), self._get_label(idx)

if __name__ == "__main__":
    # EXAMPLE USAGE for tripairs
    from framework import DATASET_PATH
    data_path = os.path.join(DATASET_PATH, "fiw", 'FIDs')
    csv_path = os.path.join(DATASET_PATH, "fiw", "tripairs", '5_cross_val.csv')
    fiw_dataset = FIW(data_path, csv_path, folds=[1,2,3,4])
    fam_list, label = fiw_dataset.__getitem__(128)
    print('is_fam = ', label)
    for fam in fam_list:
        plt.imshow(fam)
        plt.show()
