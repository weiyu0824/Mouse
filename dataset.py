import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, annotation_file='', data_dir='', train=True):
        self.annotations = pd.read_csv(annotation_file)
        self.data_dir = data_dir

        if train:
            self.annotations = self.annotations.iloc[:800000]
        else:
            self.annotations = self.annotations.iloc[800000:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_path = self.data_dir + row['file_name'] + '.npy'
        img = np.expand_dims(np.load(file_path), axis=0)
        time_id = row['time_id']
        gene_id = row['gene_id']
        slice_id = row['slice_id']
        return img, time_id, gene_id, slice_id