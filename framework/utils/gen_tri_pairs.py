import os
import random
import numpy as np
from framework import DATASET_PATH
from sklearn.model_selection import KFold
import pandas as pd
from pprint import pprint

def generate_kfold_tripairs(n_folds: int, filepath: str):
    """
    n_folds: number of folds
    filepath: Full filepath to the csv you would like to split up
             should be in fiw/tripairs/xxx-faces.csv
    """
    kf = KFold(n_splits=n_folds, shuffle=True)
    family_tripairs = pd.read_csv(filepath)

    family_ids = list(set([fid[:5] for fid in list(family_tripairs.F)]))
    kfold_ids = kf.split(family_ids)
    type_relation = os.path.basename(filepath)[:-10]

    folds = []
    for _, sub_ids in kfold_ids:
        folds.append(np.sort(list(np.array(family_ids)[sub_ids])))
    
    # DF of different folds
    df_splits = []
    for i, fold in enumerate(folds, 1):
        split = []
        for family_id in list(fold):
            ids = [i for i, s in enumerate(family_tripairs.F) if (family_id) in s]
            temp = family_tripairs.loc[ids]
            sub_face_tripairs = pd.DataFrame()
            sub_face_tripairs['F'] = temp.F
            sub_face_tripairs['M'] = temp.M
            sub_face_tripairs['C'] = temp[temp.columns[2]]
            sub_face_tripairs['fid'] = family_id
            split.append(sub_face_tripairs)
        split_df = pd.concat(split)
        labels = [1] * len(split_df)
        split_df['labels'] = labels
        
        # Add non-relations
        split_df = add_non_related_tripairs(split_df, len(split_df), family_ids)
        
        fold_ids = [i] * len(split_df)
        split_df['fold'] = fold_ids
        split_df['type'] = type_relation
        df_splits.append(split_df)

    return pd.concat(df_splits, ignore_index=True)

def add_non_related_tripairs(dataframe: pd.DataFrame, amount: int, family_ids: list):
    # For each row in dataframe create a new row with same f, m, but incorrect child
    dataframe_dict = {dataframe.columns[0]: [],
                      dataframe.columns[1]: [],
                      dataframe.columns[2]: [],
                      'fid': []
                      }
    for idx, row in dataframe.iterrows():
        # Copy father and mother row
        dataframe_dict[dataframe.columns[0]].append(row[dataframe.columns[0]])
        dataframe_dict[dataframe.columns[1]].append(row[dataframe.columns[1]])
        family_id = row[dataframe.columns[0]][:5]
        print('family_id = ', family_id)

        # Create a different child than current one, not same family
        child = get_non_related_child(dataframe, family_id)
        print('child = ', child)
        dataframe_dict[dataframe.columns[2]].append(child)
        dataframe_dict['fid'].append(family_id)
    
    print(dataframe.head())
    non_related_df = pd.DataFrame(dataframe_dict)
    non_related_df['labels'] = [0] * len(non_related_df)
    print(non_related_df.head())

    return pd.concat([dataframe, non_related_df], ignore_index=True)


def get_non_related_child(dataframe: pd.DataFrame, family_id: str):
    max_index = len(dataframe)
    child_fam_id = family_id
    while (child_fam_id == family_id):
        idx = random.randrange(max_index)
        child = dataframe.iloc[idx][2]
        child_fam_id = child[:5]
        print('child_id = ', child_fam_id)

    return child

if __name__ == "__main__":
    n_folds = 5
    folder_path = os.path.join(DATASET_PATH, "fiw", "tripairs")

    csv_files = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, csv_file) for csv_file in csv_files if csv_file.endswith("-faces.csv")]
    pprint(filepaths)

    final_df = None
    for filepath in filepaths:
        print(filepath)
        folded_tripairs_df = generate_kfold_tripairs(n_folds, filepath)
        # validation_df = add_non_family_relations(n_folds, folded_pairs_df)
        if final_df is None:
            final_df = folded_tripairs_df
        else:
            final_df = pd.concat([final_df, folded_tripairs_df], ignore_index=True)
    
    print(final_df.head())

    filepath = os.path.join(folder_path, f"{n_folds}_cross_val.csv")
    final_df.to_csv(filepath)