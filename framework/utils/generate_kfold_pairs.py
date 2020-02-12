import os
import numpy as np
from framework import DATASET_PATH
from sklearn.model_selection import KFold
import pandas as pd
import random


def generate_kfold_pairs(n_folds: int, filepath: str):
    """
    n_folds: number of folds
    filepath: Full filepath to the csv you would like to split up
             should be in fiw/lists/pairs/csv/xx-faces.csv
    """
    kf = KFold(n_splits=n_folds, shuffle=True)
    face_pairs = pd.read_csv(filepath)
    # print(len(face_pairs))
    face_ids = list(set([fid[:5] for fid in list(face_pairs.p1)]))
    # print(len(face_ids))
    kfold_ids = kf.split(face_ids)
    type_relation = os.path.basename(filepath)[:-10]
    print(type_relation)
    folds = []
    for _, sub_ids in kfold_ids:
        folds.append(np.sort(list(np.array(face_ids)[sub_ids])))
    
    # DF of different folds
    df_splits = []
    for i, fold in enumerate(folds, 1):
        split = []
        for family_id in list(fold):
            ids = [i for i, s in enumerate(face_pairs.p1) if (family_id) in s]
            sub_face_pairs = face_pairs.loc[ids]
            sub_face_pairs['fid'] = family_id
            split.append(sub_face_pairs)
        split_df = pd.concat(split)
        fold_ids = [i] * len(split_df)
        labels = [1] * len(split_df)
        split_df['fold'] = fold_ids
        split_df['labels'] = labels
        split_df['type'] = type_relation
        print(len(split_df))
        df_splits.append(split_df)

    return pd.concat(df_splits)

# TODO: Need to add non family pairs
def add_non_family_relations(n_folds: int, dataframe: pd.DataFrame):
    """
    This function will add non family relations as pairs and label them accordingly
    n_folds: number of folds, should match the number of folds used to generate the dataframe
    dataframe: should only contain pairs of correct matchs
    """
    kf = KFold(n_splits=n_folds, shuffle=True)
    face_pairs = pd.read_csv(filepath)
    family_ids = list(set([fid[:5] for fid in list(face_pairs.p1)]))
    # print(face_pairs)
    # print("Family IDs: " + str(family_ids))
    # We dont want duplicate wrong pairs 
    num_of_families = len(family_ids)
    for index, family in enumerate(family_ids):
        if(index < num_of_families-1):
            random_family = random.randint(index+1, num_of_families-1)
            family_images = face_pairs.loc[]
            person_ids = list(set([fid[:5] for fid in list(face_pairs.p1)]))
        # problem later families are preferred to previous families
            # print(random_family)
            # print(num_of_families)
            # print("Random: " + family_ids[random_family])
            # print("Target: " + family_ids[index])
    
    return 




if __name__ == "__main__":
    # Example usage
    n_folds = 5
    folder_path = os.path.join(DATASET_PATH, "fiw", "lists", "pairs", "csv")
    filepath = os.path.join(folder_path, "bb-faces.csv")
    folded_pairs_df = generate_kfold_pairs(n_folds, filepath)
    add_non_family_relations(n_folds, folded_pairs_df)
