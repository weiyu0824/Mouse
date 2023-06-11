import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from config import DATA3D_DIR
import time as time
from tqdm import tqdm
import random


np.random.seed(0)
num_ts = 7

def load_data(data_dir, train_fname, test_fname):
    train_annot = pd.read_csv(data_dir+train_fname) 
    test_annot = pd.read_csv(data_dir+test_fname) 
    train_imgs = []
    test_imgs = []
    for i in tqdm(range(len(train_annot)), desc='load train set'):
        img = np.load(data_dir+train_annot.iloc[i]['file_name']+'.npy')
        train_imgs.append(img)
        # train_imgs.append(np.ones((80, 144, 96)))
            
    for i in tqdm(range(len(test_annot)), desc='load test set'):
        img = np.load(data_dir+test_annot.iloc[i]['file_name']+'.npy')
        test_imgs.append(img)
        # test_imgs.append(np.one÷s((80, 144, 96)))
    
    data = {}
    data['train_annot'] = train_annot
    data['test_annot'] = test_annot
    data['train_imgs'] = train_imgs
    data['test_imgs'] = test_imgs
    return data


class BrainDataset(Dataset):
    def __init__(self, annotation_file='', data_dir='', transform=None):
        self.annotations = pd.read_csv(annotation_file)
        
        # DEBUG
        # self.annotations = self.annotations.loc[self.annotations['time_id'] > 2] 
        # 

        self.data_dir = data_dir
        self.transform = transform

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

class SeqBrainDataset(Dataset):
    def __init__(self, train_annot: pd.DataFrame, train_imgs, test_annot: pd.DataFrame=None, test_imgs=None, train=True):
        self.train_annot = train_annot
        self.test_annot = test_annot
        self.train_imgs = train_imgs
        self.test_imgs = test_imgs
        self.seq_len = num_ts
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_annot)
        else:
            return len(self.test_annot)
        
    def __getitem__(self, idx):
        if self.train:
            tar_gene = self.train_annot.iloc[idx]['gene_id']
            tar_time = self.train_annot.iloc[idx]['time_id']
            tar_img = self.train_imgs[idx]
        else:
            tar_gene = self.test_annot.iloc[idx]['gene_id']
            tar_time = self.test_annot.iloc[idx]['time_id']
            tar_img = self.test_imgs[idx]

            
        series = self.train_annot[(self.train_annot['gene_id'] == tar_gene)
                                & (self.train_annot['time_id'] != tar_time)]

        img_series = []
        time_series = []
        mask = []
        gene_series = []

        #
        for i in range(len(series)):
            img = self.train_imgs[series.iloc[i]['Unnamed: 0']]
            time = series.iloc[i]['time_id']
            img_series.append(img)
            time_series.append(time)
            mask.append(0)
            gene_series.append(tar_gene)
        #
        img_series.append(tar_img)
        time_series.append(tar_time)
        mask.append(1)
        gene_series.append(tar_gene)
        #
        for _ in range(self.seq_len - len(img_series)):
            z = np.zeros_like(img_series[0])
            img_series.append(z)
            time_series.append(0)
            mask.append(0)
            gene_series.append(0)

        # shuffle
        new_order = np.random.permutation(self.seq_len)
        img_series = [img_series[i] for i in new_order]
        time_series = [time_series[i] for i in new_order]
        mask = [mask[i] for i in new_order]
        gene_series = [gene_series[i] for i in new_order]

        return  np.array(img_series), np.array(time_series), np.array(mask), np.array(gene_series)
    
    
if __name__ == "__main__":

    d = SeqBrainDataset(train=True)
    d.__getitem__(0)
    # d = SeqBrainDataset(train=False)
    # d.__getitem__(0)


    # transform=None
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.0018], std=[0.0028])
    # ])


    # # Find mean and std:
    # data_dir = '/m-ent1/ent1/wylin6/mouse/preprocess/'
    # train_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True, transform=transform)
    # train_loader = DataLoader(train_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=12)

    # mean = 0
    # std = 0
    # num_samples = 0
    # min_pixel = 20
    # for data, _, _, _ in tqdm(train_loader):
    #     batch_size = data.shape[0]
    #     data = data.view(batch_size, 1, -1)

    #     mean += data.mean(2).sum(0)
    #     std += data.std(2).sum(0)

    #     min_pixel = min(min_pixel, torch.mean(data).data)
    #     num_samples += batch_size

    # mean /= num_samples
    # std /= num_samples
    # print(min_pixel)
    # print(mean)
    # print(std) 