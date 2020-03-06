
import os
import pandas as pd
import torch.utils.data as data

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
    def __init__(self, path_to_dataset: str, setting: str = 'train'):
        # NOTE: Expected that dataset already contains 50/50 split of relation / non-relation
        if setting == 'train':
        elif setting == 'validation':
        elif setting == 'test':
        else:
            raise RuntimeError(f'Unkown setting: {setting}')


    
    def _get_label(self, idx):
        # CHECK IF FROM THE SAME FAMILY
        if self.dataset.iloc[idx, 0] == self.dataset.iloc[idx, 2]:
            return 1
        else:
            return 0
    
    def __getitem__(self, idx):
        return self._get_pair(idx), self._get_label(idx)

if __name__ == "__main__":
    # EXAMPLE USAGE
    from framework import DATASET_PATH
    data_path = os.path.join(DATASET_PATH, "recognizing-faces-in-the-wild")
    fiw_dataset = FIW_kaggle(data_path)
    print(fiw_dataset.__getitem__(4))
