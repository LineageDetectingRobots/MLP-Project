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
    result = torch.sum(scores, dim=0)
    result = torch.max(result, dim=2)[1]
    return result.cpu().numpy()

def get_ensemble(ensemble_name: str):
    if ensemble_name == "high_score":
        return high_score
    else:
        raise RuntimeError('Unkown ensemble name {}'.format(ensemble_name))
    pass


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
            # y = data[1].to(model._device(), dtype=torch.long)
            y_hat = model(x)
            scores_list.append(y_hat)
    return torch.stack(scores_list)

def eval(pred_targets, dataset):
    targets = []
    for i in range(len(dataset)):
        y = dataset[i][1]
        targets.append(y)
    targets = np.array(targets)
    print('targets:', targets.shape)
    print('preds:', pred_targets.shape)
    auc = roc_auc_score(targets, pred_targets)
    return auc
    

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

    # TODO: Use ensemble method with test outputs, this should ouput a list of targets
    ensemble_method = get_ensemble(ensemble_name)
    pred_targets = ensemble_method(scores)
    # TODO: Evaluate the model on the ensemble solution
    auc = eval(pred_targets, test_dataset)
    print('test dataset auc = {}'.format(auc))


if __name__ == '__main__':
    # Get profile name from profile file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    profile_path = os.path.join(dir_path, "config", "profile.json")
    if not os.path.exists(profile_path):
        raise RuntimeError('Please add a profile.json file in MLP-Project/config/profile.json. \n' +
                            '       It should contain one line like this: {\"profile\": "profile_name"}')
    with open(profile_path, 'r') as f:
        profile_name = json.load(f)["profile"]
    
    # NOTE This should be a list of profiles you want to use. The models are expected to be saved
    model_profiles = ['DROP_TWO_DEC_arc', 'DROP_TWO_DEC_vgg2', 'DROP_TWO_DEC_vgg', 'DROP_TWO_DEC_sphere']
    ensemble_name = 'high_score'
    if model_profiles == []:
        raise RuntimeError('Please fill in model profiles')

    run_experiment(ensemble_name, model_profiles)
    