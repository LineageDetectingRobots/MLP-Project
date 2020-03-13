
from torch.utils.data import DataLoader

from framework import DATASET_PATH, RESULTS_PATH
from framework.data_provider import FIW
from framework.config.config_reader import ConfigReader
from framework.utils.gen_utils import get_data_loader

def check_network_name(network_name: str):
    networks = ['arc_face', 'vgg_face', 'sphere_face', 'vgg_face2']
    if network_name not in networks:
        raise RuntimeError(f'Unkown network name {network_name}')

def get_model(model_settings: dict):
    model_name = model_settings['model_name']
    # TODO: Given model name, get a model

    raise NotImplementedError

def train(model, train_dataset, training_settings):
    batch_size = training_settings['batch_size']
    num_of_epochs = training_settings['epochs']
    # TODO: get if we are using gpu or not
    cuda = True
    
    # Put model in training mode
    model.train()
    
    progress = tqdm.tqdm(range(1, num_of_epochs + 1))
    # TODO: We might need to drop the last element
    dataloader = get_data_loader(train_dataset, batch_size, cuda)
    for epoch in range(1, num_of_epochs + 1):
        for i, train_data in enumerate(dataloader, 1):
            stats = model.train_a_batch(train_data)
        
        # TODO: Could run on test set to see how it is learning, get some stats
        progress.update(1)

def get_datasets(network_name: str):
    # NOTE: Netowrk name is required to known which network produces the feature vectors
    csv_path = os.path.join(DATASET_PATH, 'fiw', 'tripairs', f'{network_name}_5_cross_val.csv')
    mappings_path = os.path.join(RESULTS_PATH, f'mappings_{network_name}.pickle')
    train_folds = [1, 2, 3, 4]
    test_folds = [5]
    train_dataset = FIW(csv_path, mappings_path, train_folds) 
    test_dataset = FIW(csv_path, mappings_path, test_folds)
    return train_dataset, test_dataset

def eval(model, test_dataset):
    # Evals the model based on test set to produce metrics for use
    
    model.eval()

    # TODO: get if we are using gpu or not
    cuda = True
    dataloader = get_data_loader(test_dataset, 1, cuda)

    for i, data in enumerate(dataloader, 1):
        x = data[0]
        y = data[1]

        # y_hat should be predictions 1 or 0
        y_hat = model(x)
    raise NotImplementedError

def run_experiment(profile_name: str):
    # TODO: Get configurations, from config file or something
    config_data = ConfigReader(profile_name).config_data

    # Get feature vector data
    network_name = config_data['data_settings']['network_name']
    check_network_name(network_name)
    train_dataset, test_dataset = get_datasets(network_name)
    
    # Get the model we are going to use for training
    model_settings = config_data['model_settings']
    model = get_model(model_settings)
    
    # Train the model
    training_settings = config_data['training_settings']
    train(model, train_dataset, training_settings)

    # Evaluate the model on test dataset
    eval(model, test_dataset)

    # TODO: Save results or model at the end

if __name__ == '__main__':
    profile_name = 'DEFAULT'
    run_experiment(profile_name)