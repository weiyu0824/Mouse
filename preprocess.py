import numpy as np
import pandas as pd
import torchio as tio
from typing import Tuple
import time
from torchvision import transforms
from tqdm import tqdm


# Meta Information
TIMESTAMPS = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
SHAPES = [(40, 75, 70), (69, 109, 89), (65, 132, 94), (40, 43, 67), (50, 43, 77), (50, 40, 68), (58, 41, 67)]

DATASET_DIR = '/data/wylin6/mouse/dataset/'
SAVE_DIR = '/data/wylin6/mouse/preprocess/'
IMG_SHAPE = (60, 96, 80)

def preprocess(dataset_dir: str, save_dir: str, target_shape: Tuple[int, int, int]):
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
            for slice_id in range(target_shape[0]):
                file_name = f'{ts}_{gene_id}_{slice_id}'
                annotations.append((file_name, time_id, gene_id, slice_id))

                # Save the image as .npy file
                np.save(f'{save_dir}{file_name}', img_3d[slice_id])


    df = pd.DataFrame(annotations, columns=['file_name', 'time_id', 'gene_id', 'slice_id']) 

    print(f'Save preprocessed dataset to {save_dir}')
    df.to_csv(f'{save_dir}annotation.csv')

if __name__ == "__main__":
    st = time.time()
    preprocess(DATASET_DIR, SAVE_DIR, IMG_SHAPE)
    print(time.time() - st)
