import os

from .g5nr import G5NRFlows, G5NRXBatcherFlows

def get_dataset(dataset_name, data_path, size=512, scale_factor=None, frames=2, levels=None, years=None):
    # Load dataset
    if dataset_name in ['gmao_osse_7km', 'g5nr', 'g5nr_7km']:
        dataset_train = G5NRFlows(os.path.join(data_path, 'train'),
                                  size=size, scale_factor=scale_factor,
                                  augment=True,
                                 frames=frames)
        dataset_valid = G5NRFlows(os.path.join(data_path, 'valid'),
                                  size=size, scale_factor=scale_factor,
                                 frames=frames)
    elif dataset_name == 'g5nr_xbatcher':
        dataset_train = G5NRXBatcherFlows(data_path, mode='train',
                                          size=size, frames=frames,
                                          scale_factor=scale_factor,
                                          levels=levels, years=years)
        dataset_valid = G5NRXBatcherFlows(data_path, mode='valid',
                                          size=size, frames=frames,
                                          scale_factor=scale_factor,
                                          levels=levels, years=years)
    elif dataset_name.lower() in ['goes', 'goes16']:
        # Load Geostationary Observations
        from .geostationary import L1bPatches
        dataset_train = L1bPatches(data_path, time_step=10, # minutes
                                  size=size, mode='train')
        dataset_valid = L1bPatches(data_path, time_step=10,
                                  size=size, mode='valid')

    return dataset_train, dataset_valid
