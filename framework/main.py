import os
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
from torch.utils.data import DataLoader

from framework import DATASET_PATH, RESULTS_PATH
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


def train(model, train_dataset, training_settings, validation_dataset = None):
    batch_size = training_settings['batch_size']
    num_of_epochs = training_settings['epochs']
    use_early_stopping = training_settings['early_stopping']
    if use_early_stopping and validation_dataset is not None:
        wait_epochs = training_settings['es_epochs']
        max_val_acc = -1
        es_count = 0
    
    cuda = model._is_on_cuda()

    # Put model in training mode
    model.train()
    
    progress = tqdm.tqdm(range(1, num_of_epochs + 1))
    # TODO: We might need to drop the last element
    dataloader = get_data_loader(train_dataset, batch_size, cuda, drop_last=True)
    for epoch in range(1, num_of_epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        for i, train_data in enumerate(dataloader, 1):
            # Get batch and a features length
            x = train_data[0].to(model._device(), dtype=torch.float)
            y = train_data[1].to(model._device(), dtype=torch.long)
            stats = model.train_a_batch(x, y)

            epoch_loss += stats['loss']
            epoch_acc += stats['acc']

        print('Training loss = {}'.format(epoch_loss/i))
        print('Training acc = {}'.format(epoch_acc/i))

        if validation_dataset is not None:
            acc = eval(model, validation_dataset)
            print('Validation acc = {}'.format(acc))
            # Used for early stopping
            if use_early_stopping:
                if acc >= max_val_acc:
                    max_val_acc = acc
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
    progress.close()

def get_datasets(network_name: str):
    # NOTE: Netowrk name is required to known which network produces the feature vectors
    csv_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs', f'{network_name}_5_cross_val.csv')
    mappings_path = os.path.join(RESULTS_PATH, f'mappings_{network_name}.pickle')
    train_folds = [1, 2, 3]
    validation_folds = [4]
    test_folds = [5]
    train_dataset = FIW(csv_path, mappings_path, train_folds)
    validation_dataset = FIW(csv_path, mappings_path, validation_folds)
    test_dataset = FIW(csv_path, mappings_path, test_folds)
    vec_length = list(train_dataset.__getitem__(0)[0].size())[0]
    return train_dataset, validation_dataset, test_dataset, vec_length

def eval(model, test_dataset):
    # Evals the model based on test set to produce metrics for use
    model.eval()

    # TODO: get if we are using gpu or not
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
    acc = roc_auc_score(np.array(ys), np.array(y_hats))
    return acc

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
    model_settings['input_size'] = vec_length
    model = get_model(model_settings).to(device)
    
    # Train the model
    training_settings = config_data['training_settings']
    train(model, train_dataset, training_settings, validation_dataset)

    # Evaluate the model on test dataset
    result = eval(model, test_dataset)
    print('test_result = {}'.format(result))

    # TODO: Save results or model at the end

if __name__ == '__main__':
    profile_name = 'Jack'
    run_experiment(profile_name)