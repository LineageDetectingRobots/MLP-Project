import os
import json
from typing import List
import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from framework import MODEL_PATH
from framework.utils.experiment_utils import get_model, check_network_name, get_datasets, get_data_loader
from framework.config.config_reader import ConfigReader

def high_score(scores):
    # 4, test_size, 1, 2
    result = torch.sum(scores, dim=0)
    # test_size, 1, 2
    result = torch.max(result, dim=2)[1]
    return result.cpu().numpy()

def majority_vote(scores):
    n_models = list(scores.size())[0]
    results = scores.cpu().numpy()
    result = np.argmax(results, axis=3)
    result = np.sum(result, axis=0)
    result = result/ n_models
    # Note: Not >= because if it is 50/50 more likely to be not related     
    result = (result > 0.5).astype(int)
    return result

def maj_vote_cascade(scores):
    result = np.argmax(scores, axis=2)
    result = sum(result)/len(result)
    result = int(round(result[0]))
    return result

def cascade_classifier(scores):
    pred_targets = []
    scores = scores.cpu().numpy()
    # Iterate over batch size
    for i in range(scores.shape[1]):


        # NOTE: Assuming order best model to worst model
        # Iterate over models
        pred_target = 0
        for j in range(scores.shape[0]):
            # Get score for specific model
            pred_score = scores[j, i]
            # TODO: Pick a threshold
            threshold = 0.98
            if pred_score[0][1] > threshold:
                pred_target = 1
                break
        
        # If not relation then do majority vote
        if pred_target == 0:
            j_scores = np.array([scores[j, i] for j in range(scores.shape[0])])
            pred_target = maj_vote_cascade(j_scores)
        pred_targets.append(pred_target)

    return np.array(pred_targets)

# TODO: weighted classifiers
def get_ensemble(ensemble_name: str):
    # TODO: Add more ensemble methods
    if ensemble_name == "high_score":
        return high_score
    elif ensemble_name == 'majority_vote':
        return majority_vote
    elif ensemble_name == 'cascade':
        return cascade_classifier
    else:
        raise RuntimeError('Unkown ensemble name {}'.format(ensemble_name))


def load_model(model_path, model_settings):
    model = get_model(model_settings)
    model_path = os.path.join(MODEL_PATH, model_path + '.pt')
    if not os.path.exists(model_path):
        raise RuntimeError('Unkown {} model path'.format(model_path))
    model.load_state_dict(torch.load(model_path))
    return model

def get_scores(model, dataset):
    model.eval()
    cuda = model._is_on_cuda()
    dataloader = get_data_loader(dataset, 1, cuda)

    scores_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i][0].to(model._device(), dtype=torch.float).view(1, -1)
            y_hat = model(x)
            scores_list.append(y_hat)
    return torch.stack(scores_list)

def eval(pred_targets, dataset):
    targets = []
    targets_fmd = []
    targets_fms = []
    pred_targets_fmd = []
    pred_targets_fms = []
    for i in range(len(dataset)):
        y = dataset[i][1]
        targets.append(y)
        
        type_relation = dataset.dataset.iloc[i].type
        tripair = dataset._get_tripair(i)
        
        if type_relation == 'fmd':
            targets_fmd.append(y)
            pred_targets_fmd.append(pred_targets[i])
        elif type_relation == 'fms':
            targets_fms.append(y)
            pred_targets_fms.append(pred_targets[i])
        else:
            raise RuntimeError('Unkown relationship type = {}'.format(type_relation))
    
    
    auc = roc_auc_score(np.array(targets), pred_targets)
    fmd_auc = roc_auc_score(targets_fmd, pred_targets_fmd)
    fms_auc = roc_auc_score(targets_fms, pred_targets_fms)
    return {'auc': auc,
            'fmd_auc': fmd_auc,
            'fms_auc': fms_auc}
    

def run_experiment(ensemble_name: str, model_profiles: List[str]):
    models = []
    test_datasets = []
    for profile in model_profiles:
        config_data = ConfigReader(profile).config_data

        # Get test datasets for each model
        network_name = config_data['data_settings']['network_name']
        check_network_name(network_name)
        train_dataset, validation_dataset, test_dataset, vec_length = get_datasets(network_name)
        test_datasets.append(test_dataset)

        model_settings = config_data['model_settings']
        model_settings['input_size'] = vec_length
        experiment_name = config_data['experiment_name']
        models.append(load_model(experiment_name, model_settings))

    # A list of lists of scores
    # NOTE: scores[0] -> scores of model 0 in models list 
    scores = torch.stack([get_scores(model, test_dataset) for model, test_dataset in zip(models, test_datasets)])

    # Get the ensemble method function and run it to get predicted targets
    ensemble_method = get_ensemble(ensemble_name)
    pred_targets = ensemble_method(scores)

    #  Evaluate model
    auc = eval(pred_targets, test_dataset)
    print('test dataset auc = {}'.format(auc))


if __name__ == '__main__':
    # NOTE This should be a list of profiles you want to use. The models are expected to be saved
    model_profiles = ['DROP_TWO_DEC_arc', 'DROP_TWO_DEC_vgg2', 'DROP_TWO_DEC_vgg', 'DROP_TWO_DEC_sphere']
    ensemble_name = 'majority_vote'
    if model_profiles == []:
        raise RuntimeError('Please fill in model profiles')

    run_experiment(ensemble_name, model_profiles)
