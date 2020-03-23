import os
import tqdm
import pickle 
import json
import numpy as np

from datetime import datetime
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from framework import DATASET_PATH, RESULTS_PATH, MODEL_PATH
from framework.config.config_reader import ConfigReader
from framework.utils.gen_utils import get_data_loader
from framework.utils.experiment_utils import get_model, eval, get_datasets, check_network_name



def train(model, train_dataset, training_settings, validation_dataset = None):
    # Create training dict
    train_dict = {}

    # Get config settings
    batch_size = training_settings['batch_size']
    num_of_epochs = training_settings['epochs']

    # Set up early stopping
    use_early_stopping = training_settings['early_stopping']
    if use_early_stopping and validation_dataset is not None:
        wait_epochs = training_settings['es_epochs']
        max_val_auc = -1
        es_count = 0
    
    cuda = model._is_on_cuda()

    # Put model in training mode
    model.train()
    
    progress = tqdm.tqdm(range(1, num_of_epochs + 1))
    # TODO: We might need to drop the last element
    dataloader = get_data_loader(train_dataset, batch_size, cuda, drop_last=True)
    for epoch in range(1, num_of_epochs + 1):
        epoch_loss = 0
        epoch_auc = 0
        for i, train_data in enumerate(dataloader, 1):
            # Get batch and a features length
            x = train_data[0].to(model._device(), dtype=torch.float)
            y = train_data[1].to(model._device(), dtype=torch.long)
            stats = model.train_a_batch(x, y)

            epoch_loss += stats['loss']
            epoch_auc += stats['auc']

        print('Training loss = {}'.format(epoch_loss/i))
        print('Training auc = {}'.format(epoch_auc/i))
        train_dict[f'e_{epoch}_train_loss'] = epoch_loss.detach().cpu().numpy()/i
        train_dict[f'e_{epoch}_train_acc'] = epoch_auc/i

        if validation_dataset is not None:
            eval_dict = eval(model, validation_dataset)
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
                

            model.train()
        
        # TODO: Could run on test set to see how it is learning, get some stats
        progress.update(1)
    
    train_dict['num_epochs'] = epoch
    progress.close()
    return train_dict

def run_experiment(profile_name: str):
    # Get experiment settings
    config_data = ConfigReader(profile_name).config_data
    if config_data['save_model']:
        model_name = os.path.join(MODEL_PATH, config_data['experiment_name'] + '.pt')
        if os.path.exists(model_name):
            if not config_data['overwrite_model']:
                raise Warning("Saving model when another model already exists, {}".format(model_name))
            else:
                print('Warning: Overwriting existing model. {}'.format(model_name))

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
    model_settings['input_size'] = vec_length
    model = get_model(model_settings).to(device)
    
    # Train the model
    training_settings = config_data['training_settings']
    train_dict = train(model, train_dataset, training_settings, validation_dataset)
    results_dict['training_dict'] = train_dict

    # Evaluate the model on test dataset
    result = eval(model, test_dataset)
    results_dict['test_acc'] = result
    print('test_result = {}'.format(result))

    # Save experiment results
    if config_data['save_results']:
        experiment_name = config_data['experiment_name'] + '_' + str(datetime.now())
        experiment_path = os.path.join(RESULTS_PATH, experiment_name + '.pickle')
        with open(experiment_path, 'wb') as f:
            pickle.dump(results_dict, f)
    
    if config_data['save_model']:
        model_name = os.path.join(MODEL_PATH, config_data['experiment_name'] + '.pt')
        torch.save(model.state_dict(), model_name)

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