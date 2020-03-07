import os
import random
import numpy as np
from framework import DATASET_PATH
from sklearn.model_selection import KFold
import pandas as pd
from pprint import pprint


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
    # print(type_relation)
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
        # print(len(split_df))
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
    family_ids = list(set([fid[:5] for fid in list(dataframe.p1)]))
    # We have all the families now we want to mix them up.
    # For each family we want to get the Members do a dict

    # This will create a dictionary of all the familes, family members and member's images
    # dict of family_id->member->[pictures of member]
    family_dict = {}
    for family_id in family_ids:
        # Get indexes where the families are the same
        family_photo_ids = [idx for idx, fam in enumerate(dataframe.p1) if (family_id) in fam]
        # Get family filepath as entries
        family_df_entries = dataframe.loc[family_photo_ids].p1
        family_df_entries = family_df_entries.append(dataframe.loc[family_photo_ids].p2, ignore_index=True)
        relation_type = dataframe.loc[family_photo_ids].iloc[0].type
        fold = dataframe.loc[family_photo_ids].iloc[0].fold

        
        for entry in family_df_entries:
            entry_split = entry.split('/')
            fID = entry_split[0]
            if fID not in family_dict:
                family_dict[fID] = {}
            member = entry_split[1]
            member_photo = entry_split[2]
            if member not in family_dict[fID]:
                family_dict[fID][member] = []
                family_dict[fID][member].append(member_photo)
            else:
                if member_photo not in family_dict[fID][member]:
                    family_dict[fID][member].append(member_photo)

    # Now we want to add non-related images to the dataframe
    # We want 50/50 related, non-related so there is no bias

    len_df = len(dataframe)
    new_rows = 0
    photo_1_list = []
    photo_2_list = []
    while new_rows < len_df:
        # Get 2 different, but random families
        fam_idx_1, fam_idx_2 = get_2_diff_random_indexes(len(family_dict))
        family_keys = list(family_dict.keys())
        fam_1 = family_dict[family_keys[fam_idx_1]]
        fam_2 = family_dict[family_keys[fam_idx_2]]

        # Get random member from the family 1 and a random photo from the random member
        member_keys = list(fam_1.keys())
        mem_idx_1 = random.randrange(len(member_keys))
        mem_1 = fam_1[member_keys[mem_idx_1]]
        photo_1 = mem_1[random.randrange(len(mem_1))]
        photo_1_path = family_keys[fam_idx_1] + '/' + member_keys[mem_idx_1] + '/' + photo_1
        photo_1_list.append(photo_1_path)

        member_keys = list(fam_2.keys())
        mem_idx_2 = random.randrange(len(member_keys))
        mem_2 = fam_2[member_keys[mem_idx_2]]
        photo_2 = mem_2[random.randrange(len(mem_2))]
        photo_2_path = family_keys[fam_idx_2] + '/' + member_keys[mem_idx_2] + '/' + photo_2
        photo_2_list.append(photo_2_path)
        
        new_rows += 1
    
    # Create a df and append it to the given df
    new_df = pd.DataFrame(photo_1_list, columns=['p1'])
    new_df['p2'] = photo_2_list
    new_df['labels'] = [0] * len(new_df)

    new_df['type'] = ['bb'] * len(new_df)
    new_df['fold'] = [1] * len(new_df)
    new_df['fid'] = ['idc'] * len(new_df)
    # print(new_df)
    result_df = pd.concat([dataframe, new_df], ignore_index=True)
    # print(result_df)
    return result_df

def get_2_diff_random_indexes(max_index: int):
    idx_1 = 0
    idx_2 = 0
    while (idx_1 == idx_2):
        idx_1 = random.randrange(max_index)
        idx_2 = random.randrange(max_index)
    return idx_1, idx_2

if __name__ == "__main__":
    # Example usage
    n_folds = 5
    folder_path = os.path.join(DATASET_PATH, "fiw", "lists", "pairs", "csv")
    # filepath = os.path.join(folder_path, "bb-faces.csv")
    csv_files = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, csv_file) for csv_file in csv_files if csv_file.endswith("-faces.csv")]

    final_df = None
    for filepath in filepaths:
        print(filepath)
        folded_pairs_df = generate_kfold_pairs(n_folds, filepath)
        validation_df = add_non_family_relations(n_folds, folded_pairs_df)
        if final_df is None:
            final_df = validation_df
        else:
            final_df = pd.concat([final_df, validation_df], ignore_index=True)

    # print(final_df)

    folder_path = os.path.join(DATASET_PATH, "fiw")
    filepath = os.path.join(folder_path, "new_val.csv")
    final_df.to_csv(filepath)
