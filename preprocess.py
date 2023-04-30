import numpy as np
import pandas as pd
import torchio as tio
from typing import Tuple
import time
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# Meta Information
TIMESTAMPS = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
SHAPES = [(40, 75, 70), (69, 109, 89), (65, 132, 94), (40, 43, 67), (50, 43, 77), (50, 40, 68), (58, 41, 67)]

DATASET_DIR = '/m-ent1/ent1/wylin6/mouse/dataset/'
SAVE_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess/'
IMG_SHAPE = (80, 144, 96) #(60, 96, 80)

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

def show_mid_slice(img_numpy, title='img', save_path=None):
    """
    Accepts an 3D numpy array and shows median slices in all three planes
    :param img_numpy:
    """
    assert img_numpy.ndim == 3
    n_i, n_j, n_k = img_numpy.shape

    # saggital
    center_i1 = int((n_i - 1) / 2)
    # transverse
    center_j1 = int((n_j - 1) / 2)
    # axial slice
    center_k1 = int((n_k - 1) / 2)

    show_slices([img_numpy[center_i1, :, :],
                 img_numpy[:, center_j1, :],
                 img_numpy[:, :, center_k1]], save_path=save_path)
    plt.suptitle(title)

def show_slices(slices, save_path=None):
    """
    Function to display a row of image slices
    Input is a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T)
    if save_path != None:
        fig.savefig(save_path)

def crop_img(img):
    """
    Cropping the padding of the image
    """
    coords = np.array(np.nonzero(img))
    min_coord = np.min(coords, axis=1)
    max_coord = np.max(coords, axis=1)

    cropped_img = img[min_coord[0]:max_coord[0],
                        min_coord[1]:max_coord[1],
                        min_coord[2]:max_coord[2]]
    return cropped_img

def add_pad(img, new_shape):
    """
    Pad the image to the specific size
    """
    new_depth, new_height, new_width = new_shape
    depth, height, width = img.shape
    # print(img.shape)
    # print(width)
    final_image = np.zeros((new_depth, new_height, new_width))

    pad_front = int((new_depth - depth) // 2)
    pad_top = int((new_height - height) // 2)
    pad_left = int((new_width - width) // 2)
    # print(pad_left)
    
    # Replace the pixels with the image's pixels
    final_image[pad_front:pad_front+depth, pad_top:pad_top+height, pad_left:pad_left+width] = img
    
    return final_image

def normalize(img, min=0, max=0.3):
    """
    Clip the image to min ~ max, then scale to 0 ~ 1
    """
    norm_img = (img - min) / (max - min) 
    norm_img = np.clip(norm_img, 0, 1)
    return norm_img

def preprocess_3d(dataset_dir: str, save_dir: str, target_shape: Tuple[int, int, int]):
    # Create save dir if dir not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Stats 
    stats = {}

    annotations = []
    for time_id, (shape, ts) in tqdm(enumerate(zip(SHAPES, TIMESTAMPS)), desc='time loop'):
    # for time_id, (shape, ts) in enumerate(zip(SHAPES, TIMESTAMPS)):
        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')
        
        for img_id, gene_id in tqdm(enumerate(gene_ids), desc='gene loop'):
        # for img_id, gene_id in enumerate(gene_ids):
            
            

            img_3d = np.reshape(imgs[img_id], shape)
            
            

            # # STATS: Check if value range change according to gene
            # if gene_id not in stats.keys():
            #     stats[gene_id] = []
            # stats[gene_id].append(np.max(img_3d))

            
            # Pad all images to target size
            img_3d = crop_img(img_3d)
            img_3d = add_pad(img_3d, IMG_SHAPE)
            img_3d = normalize(img_3d)
            show_mid_slice(img_3d, save_path=f'vis/{time_id}_{gene_id}.png')
            
            break
            
            # # TODO: Resize

            # # Save image
            # file_name = f'{ts}_{gene_id}'
            # annotations.append((file_name, time_id, gene_id))
            # np.save(f'{save_dir}{file_name}', img_3d)

    df = pd.DataFrame(annotations, columns=['file_name', 'time_id', 'gene_id']) 
    print(f'Save preprocessed 3d dataset to {save_dir}')
    df.to_csv(f'{save_dir}annotation.csv')

    # print(stats)

if __name__ == "__main__":
    st = time.time()
    # preprocess_2d(DATASET_DIR, '/m-ent1/ent1/wylin6/mouse/preprocess/', IMG_SHAPE)

    preprocess_3d(DATASET_DIR, '/m-ent1/ent1/wylin6/mouse/preprocess_3d/', IMG_SHAPE)
    print(time.time() - st)
