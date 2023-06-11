import numpy as np
import pandas as pd
import torchio as tio
from typing import Tuple
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random
import argparse
from plot import plot_3d_heatmap, plot_every_slice
from config import TIMESTAMPS, SHAPES, DATASET_DIR, DATA2D_DIR, DATA3D_DIR, GUIDE_DIR, IMG_SHAPE

def gamma_encode(img, gamma=0.1, max_val=0.3, clip_min=0, clip_max=1):
    img = img ** gamma
    max_val = max_val ** gamma
    img = img / max_val
    img = np.clip(img, clip_min, clip_max)
    return img

def gamma_decode(img, gamma=0.1, max_val=0.3, clip_min=0, clip_max=1): 
    max_val = max_val ** gamma
    img = img ** (1/gamma)
    return img

def add_pad(img, new_shape):
    """
    Pad the image to the specific size
    """
    new_depth, new_height, new_width = new_shape
    depth, height, width = img.shape
    final_image = np.zeros((new_depth, new_height, new_width))

    pad_front = int((new_depth - depth) // 2)
    pad_top = int((new_height - height) // 2)
    pad_left = int((new_width - width) // 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_front:pad_front+depth, pad_top:pad_top+height, pad_left:pad_left+width] = img
    
    return final_image

def remove_pad(img, new_shape):
    new_depth, new_height, new_width = new_shape
    depth, height, width = img.shape

    pad_front = int((depth - new_depth) // 2)
    pad_top = int((height - new_height) // 2)
    pad_left = int((width - new_width) // 2)

    return img[pad_front:pad_front+new_depth, pad_top:pad_top+new_height, pad_left:pad_left+new_width]

def normalize(img, min=0, max=0.001):
    """
    Clip the image to min ~ max, then scale to 0 ~ 1
    """
    norm_img = (img - min) / (max - min) 
    norm_img = np.clip(norm_img, 0, 1)
    return norm_img

def preprocess_2d(dataset_dir: str, save_dir: str, target_shape: Tuple[int, int, int]):
    # Create save dir if dir not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform = transforms.Compose([tio.transforms.Resize(IMG_SHAPE)])

    annotations = []
    for time_id, (shape, ts) in tqdm(enumerate(zip(SHAPES, TIMESTAMPS)), desc='time loop'):
        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')
        
        for img_id, gene_id in tqdm(enumerate(gene_ids), desc='gene loop'):
            img_3d = np.reshape(imgs[img_id], shape)

            img_3d = np.expand_dims(img_3d, axis=0)
            img_3d = transform(img_3d) # transform need 4d tensor 
            img_3d = np.reshape(img_3d, IMG_SHAPE)

            # Save every slice of image seperately.
            slice_id = 0
            for i in range(target_shape[0]):
                if np.max(img_3d[i]) != 0:
                    # Save the image as .npy file
                    file_name = f'{ts}_{gene_id}_{slice_id}'
                    annotations.append((file_name, time_id, gene_id, slice_id))
                    np.save(f'{save_dir}{file_name}', img_3d[i])
                    
                    slice_id += 1



    df = pd.DataFrame(annotations, columns=['file_name', 'time_id', 'gene_id', 'slice_id']) 

    print(f'Save preprocessed dataset to {save_dir}')
    df.to_csv(f'{save_dir}annotation.csv')

def preprocess_3d(dataset_dir: str, save_dir: str, target_shape: Tuple[int, int, int], train_ratio=0.8):
    # Create save dir if dir not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    annotations = []
    for time_id, (shape, ts) in tqdm(enumerate(zip(SHAPES, TIMESTAMPS)), desc='time loop'):
    # for time_id, (shape, ts) in enumerate(zip(SHAPES, TIMESTAMPS)):
        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')
        
        for img_id, gene_id in tqdm(enumerate(gene_ids), desc='gene loop'):
        # for img_id, gene_id in enumerate(gene_ids):
            
            img_3d = np.reshape(imgs[img_id], shape)
            
            # DEBUGS
            # plot_3d_heatmap(img_3d, save_path=f'vis/3d_heat_map_{time_id}_{gene_id}.png', heat_min=0.001)
            # plot_every_slice(img_3d, save_path=f'debug_{time_id}_{gene_id}.png')
            # DEBUG END            

            # Pad all images to target size
            img_3d = add_pad(img_3d, IMG_SHAPE)
            img_3d = gamma_encode(img_3d)

            # Save image
            if np.max(img_3d) != 0:
                file_name = f'{ts}_{gene_id}'
                annotations.append((file_name, time_id, gene_id))
                np.save(f'{save_dir}{file_name}', img_3d)
            else:
                print('Remove empty brain')


    # Build Train and Test 
    random.shuffle(annotations)
    train_size = len(annotations) * train_ratio
    train_anntations = annotations[0:int(train_size)]
    test_annotations = annotations[int(train_size):]

    train_df = pd.DataFrame(train_anntations, columns=['file_name', 'time_id', 'gene_id']) 
    test_df = pd.DataFrame(test_annotations, columns=['file_name', 'time_id', 'gene_id']) 

    print(f'Save preprocessed 3d dataset to {save_dir}')
    train_df.to_csv(f'{save_dir}train_annotation.csv')
    test_df.to_csv(f'{save_dir}test_annotation.csv')

def statistics(dataset_dir):
    states = {}
    for time_id, (shape, ts) in enumerate(zip(SHAPES, TIMESTAMPS)):
        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')
        states[time_id] = imgs.flatten()

    for key in states.keys():
        print('----------')
        print(key)
        print(np.percentile(states[key], 50))
        print(np.percentile(states[key], 75))
        print(np.percentile(states[key], 85))
        print(np.percentile(states[key], 99))

def gen_guide(dataset_dir: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    threshold = 0.01
    annotations = pd.read_csv(dataset_dir+'train_annotation.csv')

    guides = [None for _ in range(7)]
    ngenes = [0 for _ in range(7)]
    for i in tqdm(range(len(annotations))):
        row = annotations.iloc[i]
        file_path = dataset_dir + row['file_name'] + '.npy'
        img = np.load(file_path)
        time_id = row['time_id']
        if guides[time_id] is None:
            guides[time_id] = gamma_decode(img)
        else:
            guides[time_id] += gamma_decode(img)
        ngenes[time_id] += 1
    for i in range(len(guides)):
        guides[i] = np.where(guides[i]>threshold*ngenes[i],guides[i]/ngenes[i], 0)
        plot_every_slice(guides[i], save_path=f'{save_dir}/guide_{i}.png')

    print(f'Save preprocessed guides to {save_dir}')
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Preprocess')
    parser.add_argument('command', choices=['2d', '3d', 'guide', 'stats'])
    parser.add_argument('-d', '--dataset_dir', default=DATASET_DIR)      # option that takes a value
    parser.add_argument('-sg', '--save_guide_dir', default=GUIDE_DIR) 
    parser.add_argument('-s3', '--save_3d_dir', default=DATA3D_DIR)
    parser.add_argument('-s2', '--save_2d_dir', default=DATA2D_DIR)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    save_guide_dir = args.save_guide_dir 
    save_3d_dir = args.save_3d_dir
    save_2d_dir = args.save_2d_dir
    
    if args.command == '2d':
        preprocess_2d(dataset_dir, save_2d_dir, IMG_SHAPE)
    elif args.command == '3d':
        preprocess_3d(dataset_dir, save_3d_dir, IMG_SHAPE)
    elif args.command == 'guide':
        gen_guide(dataset_dir, save_dir=save_guide_dir)
    elif args.command == 'stats':
        statistics(dataset_dir)
    
    
