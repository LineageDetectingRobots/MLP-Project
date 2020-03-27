
# TODO: read train vals
# TODO: combine features
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from framework.utils.experiment_utils import get_datasets
from framework import DATASET_PATH

def combine_fmc_features(father_vecs, mother_vecs, child_vecs):
    fc_cosine_sim = np.array([distance.cosine(u, v) for u, v in zip(father_vecs, child_vecs)]).reshape(-1, 1)
    mc_cosine_sim = np.array([distance.cosine(u, v) for u, v in zip(mother_vecs, child_vecs)]).reshape(-1, 1)
    fc_mc_cosine_sim = np.hstack((fc_cosine_sim, mc_cosine_sim))
    return np.mean(fc_mc_cosine_sim, axis=1)


def get_vecs_and_labels(dataset):
    # get features for father, mother and child
    father_vecs = []
    mother_vecs = []
    child_vecs = []
    labels = []
    relation_types = []
    for i in range(len(dataset)):
        fam_list = dataset.get_father_mother_child(i)
        father = fam_list[0]
        mother = fam_list[1]
        child = fam_list[2]
        father_vecs.append(father)
        mother_vecs.append(mother)
        child_vecs.append(child)
        labels.append(dataset.dataset.iloc[i].labels)
        relation_types.append(dataset.dataset.iloc[i].type)

    return father_vecs, mother_vecs, child_vecs, labels, relation_types
    


if __name__ == "__main__":
    network_name = "sphere_face"
    train_dataset, validation_dataset, test_dataset, vec_length = get_datasets(network_name)

    folder_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs')
    val_csv = os.path.join(folder_path, "5_cross_val.csv")
    val_df = pd.read_csv(val_csv)

    father_vecs, mother_vecs, child_vecs, labels, relation_types = get_vecs_and_labels(validation_dataset)

    val_scores = combine_fmc_features(father_vecs, mother_vecs, child_vecs)
    val_labels = labels

    # Find best threshold
    thresholds = np.arange(1, 0, step=-0.0125)
    accuracy_scores = []
    for thresh in tqdm(thresholds):
        accuracy_scores.append(roc_auc_score(val_labels, val_scores > thresh))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max() 
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    print(f"Max accuracy: {max_accuracy}")
    print(f"Max accuracy threshold: {max_accuracy_threshold}")

    # eval on test dataset
    father_vecs, mother_vecs, child_vecs, labels, relation_types = get_vecs_and_labels(test_dataset)

    test_scores = combine_fmc_features(father_vecs, mother_vecs, child_vecs)
    test_labels = labels

    preds = test_scores > max_accuracy_threshold

    
    for type in np.unique(np.array(relation_types)):
        pred_labels_type = []
        labels_type = []
        for i in range(len(preds.astype(int))):
            if type == relation_types[i]:
                pred_labels_type.append(preds[i])
                labels_type.append(test_labels[i])
        
        print('score {}: {}'.format(type, roc_auc_score(labels_type, pred_labels_type)))

    # print(preds.astype(int))
    # print(accuracy_score(test_labels, preds))

    

