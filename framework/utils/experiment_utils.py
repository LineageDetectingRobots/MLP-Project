import os
import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from framework import DATASET_PATH, RESULTS_PATH
from framework.networks.linear_layer import MLP
from framework.utils.gen_utils import get_data_loader
from framework.data_provider import FIW

def check_network_name(network_name: str):
    networks = ['arc_face', 'vgg_face', 'sphere_face', 'vgg_face2']
    if network_name not in networks:
        raise RuntimeError(f'Unkown network name {network_name}')

def get_datasets(network_name: str):
    # NOTE: Netowrk name is required to known which network produces the feature vectors
    csv_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs', '5_cross_val.csv')
    mappings_path = os.path.join(RESULTS_PATH, f'mappings_{network_name}.pickle')
    train_folds = [1, 2, 3]
    validation_folds = [4]
    test_folds = [5]
    train_dataset = FIW(csv_path, mappings_path, train_folds)
    validation_dataset = FIW(csv_path, mappings_path, validation_folds)
    test_dataset = FIW(csv_path, mappings_path, test_folds)
    vec_length = list(train_dataset.__getitem__(0)[0].size())[0]
    return train_dataset, validation_dataset, test_dataset, vec_length

def get_model(model_settings: dict):
    model_name = model_settings['model_name']
    model = None
    
    # Implement for other models
    if model_name == 'MLP':
        model = MLP(model_settings)
    else:
        raise RuntimeError(f'unkown model name = {model_name}')

    return model

def eval(model, test_dataset):
    # Evals the model based on test set to produce metrics for use
    model.eval()

    cuda = model._is_on_cuda()
    dataloader = get_data_loader(test_dataset, 1, cuda)
    y_hats = []
    ys = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 1):
            # Get batch and a features length
            x = data[0].to(model._device(), dtype=torch.float)
            y = data[1].to(model._device(), dtype=torch.long)

            y_hat = model(x)
            pred_target = torch.max(y_hat, dim=1)[1]
            y_hats.append(pred_target.cpu().numpy())
            ys.append(y.cpu().numpy())
    auc = roc_auc_score(np.array(ys), np.array(y_hats))


    y_hats_fmd = []
    y_hats_fms = []
    ys_fmd = []
    ys_fms = []
    for idx in range(len(test_dataset.dataset)):
        # Get type
        type_relation = test_dataset.dataset.iloc[idx].type
        tripair = test_dataset._get_tripair(idx)
        label =  test_dataset._get_label(idx)

        x = tripair.to(model._device(), dtype=torch.float).view(1, -1)
        y = label

        y_hat = model(x)
        pred_target = torch.max(y_hat, dim=1)[1]
        if type_relation == 'fmd':
            y_hats_fmd.append(pred_target.cpu().numpy())
            ys_fmd.append(y)
        elif type_relation == 'fms':
            y_hats_fms.append(pred_target.cpu().numpy())
            ys_fms.append(y)
        else:
            raise RuntimeError('Unkown relationship type = {}'.format(type_relation))
    fmd_auc = roc_auc_score(np.array(ys_fmd), np.array(y_hats_fmd))
    fms_auc = roc_auc_score(np.array(ys_fms), np.array(y_hats_fms))
    return {'auc': auc,
            'fmd_auc': fmd_auc,
            'fms_auc': fms_auc}