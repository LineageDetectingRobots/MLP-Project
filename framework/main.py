from torch.utils.data import DataLoader

def get_model(model_name; str):
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

def get_datasets(model_name):
    # Get training csv, feature vecs and conversion from filepath to feature_vec
    raise NotImplementedError

def eval():
    # Evals the model based on test set to produce metrics for use
    raise NotImplementedError

def run_experiment():
    # TODO: Get configurations, from config file or something
    
    # TODO: Load dataset
    train_dataset, test_dataset = get_datasets(model_name)
    
    # TODO: get model we want to use
        # linear layers
    model = get_model(model_name)
    
    # TODO: Train
    train(model, num_of_epochs, train_dataset, cuda, batch_size=32)

    # TODO: Eval
    eval()


if __name__ == '__main__':
    run_experiment()