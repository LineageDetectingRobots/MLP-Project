
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=default_collate, drop_last=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''
    # TODO(Jack): Add augmentation of dataset
    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=drop_last,
        **({'num_workers': 4, 'pin_memory': True} if cuda else {})
    )