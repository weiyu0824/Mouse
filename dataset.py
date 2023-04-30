import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch

class BrainDataset(Dataset):
    def __init__(self, annotation_file='', data_dir='', train=True, transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.data_dir = data_dir
        self.transform = transform

        if train:
            self.annotations = self.annotations.iloc[:800000]
        else:
            self.annotations = self.annotations.iloc[800000:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_path = self.data_dir + row['file_name'] + '.npy'
        # img = torch.tensor(np.expand_dims(np.load(file_path), axis=0))
        img = torch.tensor(np.load(file_path))
        time_id = torch.tensor(row['time_id'])
        gene_id = torch.tensor(row['gene_id'])

        # slice_id = torch.tensor(row['slice_id'])
        if self.transform:
            img = self.transform(img)

        # 2d data
        # return img, time_id, gene_id, slice_id
        # 3d data
        return img, time_id, gene_id

if __name__ == "__main__":
    transform=None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0018], std=[0.0028])
    ])


    # Find mean and std:
    data_dir = '/m-ent1/ent1/wylin6/mouse/preprocess/'
    train_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=12)

    mean = 0
    std = 0
    num_samples = 0
    min_pixel = 20
    for data, _, _, _ in tqdm(train_loader):
        batch_size = data.shape[0]
        data = data.view(batch_size, 1, -1)

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

        min_pixel = min(min_pixel, torch.mean(data).data)
        num_samples += batch_size

    mean /= num_samples
    std /= num_samples
    print(min_pixel)
    print(mean)
    print(std) 