from framework import DATASET_PATH, RESULTS_PATH
from torch.utils.data import DataLoader
from framework.data_provider import FIW

def check_network_name(network_name: str):
    networks = []
    if network_name not in networks:
        raise RuntimeError(f'Unkown network name {network_name}')

def get_model(model_name: str):
    raise NotImplementedError

def train(model, num_of_epochs, train_dataset, cuda, batch_size = 64):
    model.train()
    
    progress = tqdm.tqdm(range(1, num_of_epochs+1))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True,
                            **({'num_workers': 4, 'pin_memory': True} if cuda else {})
    for epoch in range(1, self.num_of_epochs+1):
        
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

def eval(model):
    # Evals the model based on test set to produce metrics for use
    raise NotImplementedError

def run_experiment():
    # TODO: Get configurations, from config file or something
    network_name = 'something'
    check_network_name(network_name)
    
    train_dataset, test_dataset = get_datasets(network_name: str)
    
    # TODO: get model we want to use
        # linear layers
    model = get_model(model_name)
    
    # TODO: Train
    train(model, num_of_epochs, train_dataset, cuda, batch_size=32)

    # TODO: Eval
    eval(model)


if __name__ == '__main__':
    run_experiment()