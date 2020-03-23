import os
import tqdm
import pickle 
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from framework import DATASET_PATH, RESULTS_PATH, MODEL_PATH
from framework.networks.linear_layer import MLP
from framework.data_provider import FIW
from framework.config.config_reader import ConfigReader
from framework.utils.gen_utils import get_data_loader

def check_network_name(network_name: str):
    networks = ['arc_face', 'vgg_face', 'sphere_face', 'vgg_face2']
    if network_name not in networks:
        raise RuntimeError(f'Unkown network name {network_name}')

def get_model(model_settings: dict):
    model_name = model_settings['model_name']
    model = None
    
    # Implement for other models
    if model_name == 'MLP':
        model = MLP(model_settings)
    else:
        raise RuntimeError(f'unkown model name = {model_name}')

    return model


def train(model_1, model_2, train_dataset, training_settings, validation_dataset = None):
    # Create training dict
    train_dict = {}
    checkpoint = training_settings['checkpoint']

    # Get config settings
    batch_size = training_settings['batch_size']
    num_of_epochs = training_settings['epochs']

    # Set up early stopping
    use_early_stopping = training_settings['early_stopping']
    if use_early_stopping and validation_dataset is not None:
        wait_epochs = training_settings['es_epochs']
        max_val_auc = -1
        es_count = 0
    
    cuda = model_1._is_on_cuda()

    # Put model in training mode
    model_1.train()
    model_2.train()
    
    progress = tqdm.tqdm(range(1, num_of_epochs + 1))
    # TODO: We might need to drop the last element
    dataloader = get_data_loader(train_dataset, batch_size, cuda, drop_last=True)
    for epoch in range(1, num_of_epochs + 1):
        epoch_loss = [0, 0]
        epoch_auc = [0, 0]
        for i, train_data in enumerate(dataloader, 1):
            # Get batch and a features length
            vec_size = list(train_data[0].size())[1] // 3
            father = train_data[0][:, 0:vec_size]
            mother = train_data[0][:, vec_size:vec_size*2]
            child = train_data[0][:, vec_size*2:vec_size*3]

            father_child = torch.cat((father, child), 1)
            mother_child = torch.cat((mother, child), 1)
            x_1 = father_child.to(model_1._device(), dtype=torch.float)
            x_2 = mother_child.to(model_2._device(), dtype=torch.float)
            y = train_data[1].to(model_1._device(), dtype=torch.long)

            stats_1 = model_1.train_a_batch(x_1, y)
            stats_2 = model_2.train_a_batch(x_2, y)

            epoch_loss[0] += stats_1['loss']
            epoch_loss[1] += stats_2['loss']

            epoch_auc[0] += stats_1['auc']
            epoch_auc[1] += stats_2['auc']

        print('Training loss father_child = {}'.format(epoch_loss[0]/i))
        print('Training loss mother_child = {}'.format(epoch_loss[1]/i))
        print('Training auc father_child = {}'.format(epoch_auc[0]/i))
        print('Training auc mother_child = {}'.format(epoch_auc[1]/i))
        train_dict[f'e_{epoch}_train_loss_1'] = epoch_loss[0].detach().cpu().numpy()/i
        train_dict[f'e_{epoch}_train_loss_2'] = epoch_loss[1].detach().cpu().numpy()/i
        train_dict[f'e_{epoch}_train_acc_1'] = epoch_auc[0]/i
        train_dict[f'e_{epoch}_train_acc_2'] = epoch_auc[1]/i

        if validation_dataset is not None and epoch % checkpoint == 0:
            eval_dict = eval(model_1, model_2, validation_dataset)
            train_dict[f'e_{epoch}_val_auc'] = eval_dict['auc']
            print('Validation dict = {}'.format(eval_dict))
            # Used for early stopping
            if use_early_stopping:
                if eval_dict['auc'] >= max_val_auc:
                    max_val_auc = eval_dict['auc']
                    es_count = 0
                else:
                    es_count += 1

                # If val acc has not increased for x epochs then stop training 
                if wait_epochs < es_count:
                    progress.close()
                    print('Early stopping....')
                    break
                

            model_1.train()
            model_2.train()
        
        # TODO: Could run on test set to see how it is learning, get some stats
        progress.update(1)
    
    train_dict['num_epochs'] = epoch
    progress.close()
    return train_dict

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

def eval(model_1, model_2, test_dataset):
    # Evals the model based on test set to produce metrics for use
    model_1.eval()
    model_2.eval()

    cuda = model_1._is_on_cuda()
    dataloader = get_data_loader(test_dataset, 1, cuda)
    y_hats = []
    ys = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 1):
            # Get batch and a features length
            vec_size = list(data[0].size())[1] // 3
            father = data[0][:, 0:vec_size]
            mother = data[0][:, vec_size:vec_size*2]
            child = data[0][:, vec_size*2:vec_size*3]
  
            x_1 = torch.cat((father, child), 1).to(model_1._device(), dtype=torch.float)
            x_2 = torch.cat((mother, child), 1).to(model_2._device(), dtype=torch.float)
            y = data[1].to(model_1._device(), dtype=torch.long)

            y_hat_1 = model_1(x_1)
            y_hat_2 = model_2(x_2)
            y_hat = y_hat_1 + y_hat_2
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
        vec_size = list(tripair.size())[0] // 3

        father = tripair[0:vec_size]
        mother = tripair[vec_size:vec_size*2]
        child = tripair[vec_size*2:vec_size*3]

        x_1 = torch.cat((father, child)).to(model_1._device(), dtype=torch.float)
        x_2 = torch.cat((mother, child)).to(model_2._device(), dtype=torch.float)
        y = label

        y_hat_1 = model_1(x_1.view(1, -1))
        y_hat_2 = model_2(x_2.view(1, -1))
        y_hat = y_hat_1 + y_hat_2
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

def run_experiment(profile_name: str):
    # Get experiment settings
    config_data = ConfigReader(profile_name).config_data

    results_dict = {}
    results_dict['config_data'] = config_data

    use_cuda = config_data["use_cuda"]
    cuda = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Get feature vector data
    network_name = config_data['data_settings']['network_name']
    check_network_name(network_name)
    train_dataset, validation_dataset, test_dataset, vec_length = get_datasets(network_name)
    
    # Get the model we are going to use for training
    model_settings = config_data['model_settings']
    model_settings['input_size'] = (vec_length // 3) * 2
    model_1 = get_model(model_settings).to(device)
    model_2 = get_model(model_settings).to(device)
    
    # Train the model
    training_settings = config_data['training_settings']
    train_dict = train(model_1, model_2, train_dataset, training_settings, validation_dataset)
    results_dict['training_dict'] = train_dict

    # Evaluate the model on test dataset
    result = eval(model_1, model_2, test_dataset)
    results_dict['test_acc'] = result
    print('test_result = {}'.format(result))

    # TODO: Maybe save model
    # Save experiment results
    if config_data['save_results']:
        experiment_name = config_data['experiment_name'] + '_' + str(datetime.now())
        experiment_path = os.path.join(RESULTS_PATH, experiment_name)
        with open(experiment_path, 'wb') as f:
            pickle.dump(results_dict, f)
    
    if config_data['save_model']:
        model_name_1 = os.path.join(MODEL_PATH, config_data['experiment_name'] + '_fc')
        model_name_2 = os.path.join(MODEL_PATH, config_data['experiment_name'] + '_mc')
        model_1.save_state_dict(model_name_1)
        model_2.save_state_dict(model_name_2)
    pprint(results_dict)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    profile_path = os.path.join(dir_path, "config", "profile.json")
    if not os.path.exists(profile_path):
        raise RuntimeError('Please add a profile.json file in MLP-Project/config/profile.json. \n' +
                            '       It should contain one line like this: {\"profile\": "profile_name"}')
    with open(profile_path, 'r') as f:
        profile_name = json.load(f)["profile"]
    run_experiment(profile_name)