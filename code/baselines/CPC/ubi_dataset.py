import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader





# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, args, phase):
        
        self.filename = args.data_file
        self.dataset = args.dataset
        self.data_raw = self.load_dataset(self.dataset,self.filename)
        assert args.input_size == self.data_raw[phase]['data'].shape[2]

        # Obtaining the segmented data
        self.data, self.labels = self.data_raw[phase]['data'], self.data_raw[phase]['labels']
        
    def load_dataset(self, dataset, filename):
        data_raw = {'train': {'data': np.load(os.path.join(filename, 'train_x.npy')),
                              'labels': np.load(os.path.join(filename, 'train_y.npy'))[:, 1]},
                    'val': {'data': np.load(os.path.join(filename, 'val_x.npy')),
                            'labels': np.load(os.path.join(filename, 'val_y.npy'))[:, 1]},
                    'test': {'data': np.load(os.path.join(filename, 'test_x.npy')),
                             'labels': np.load(os.path.join(filename, 'test_y.npy'))[:, 1]}}
        if dataset=='MESA':
            for set in ['train', 'val', 'test']:
                data_raw[set]['data'] = np.transpose(data_raw[set]['data'], (0, 2, 1))
                data_raw[set]['data'] = data_raw[set]['data'].astype(np.float32)
                data_raw[set]['labels'] = data_raw[set]['labels'].astype(np.uint8)

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


def load_dataset(args, classifier=False):
    datasets = {x: HARDataset(args=args, phase=x) for x in
                ['train', 'val', 'test']}

    def get_batch_size():
        if classifier:
            batch_size = args.classifier_batch_size
        else:
            batch_size = args.batch_size

        return batch_size

    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=get_batch_size(),
                                  shuffle=True if x == 'train' else False,
                                  #num_workers=2, pin_memory=True
                                  )
                    for x in ['train', 'val', 'test']}

    # Printing the batch sizes
    for phase in ['train', 'val', 'test']:
        print('The batch size for {} phase is: {}'
              .format(phase, data_loaders[phase].batch_size))

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)

    return data_loaders, dataset_sizes


