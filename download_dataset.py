import os
import shutil
import pandas as pd 
import numpy as np
import argparse
import zipfile
import urllib.request as urllibreq
import xml.dom.minidom as xmldom
from multiprocessing import Pool
import time
from config import QUERY_FILE_PATH, TIMESTAMPS

# Setting 
query_file_path = QUERY_FILE_PATH

# Output file directories
DATASET_DIR = '/m-ent1/ent1/wylin6/mouse/dataset/'

# URL helper func
def get_xml_url(timestamp, gene_sym):
    url = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model" \
            "::SectionDataSet,rma::criteria,[failed$eq%27false%27],genes[acronym$eq%27" \
            f"{gene_sym}%27],products[abbreviation$eq%27DevMouse%27],specimen(donor(age[name$eq%27" \
            f"{timestamp}%27]))"
    return url

def get_grid_url(exp_id):
    return f"http://api.brain-map.org/grid_data/download/{exp_id}?include=density"


def download_data(timestamp, dataset_dir):
    print(f'Downloading {timestamp} data')

    tmp_cache_dir = dataset_dir + timestamp + '/'
    tmp_xml_save_path = tmp_cache_dir + 'query.xml'
    tmp_zip_save_path = tmp_cache_dir + 'grid.zip'

    # Create dir if dir not exist
    if not os.path.exists(tmp_cache_dir):
        os.makedirs(tmp_cache_dir)

    meta_save_path = dataset_dir + timestamp + '.mhd'
    imgs_save_path = dataset_dir + timestamp + '.npy'
    genes_save_path = dataset_dir + timestamp + '.genes.npy'
    imgs = []
    gene_ids = []

    for id, gene_sym in enumerate(gene_symbols):
        print(f'Downloading {gene_sym}({id}) at {timestamp}', flush=True)
        
        try: 
            # Use xml file to get the experiment ID
            url_xml = get_xml_url(timestamp, gene_sym)
            urllibreq.urlretrieve(url_xml, tmp_xml_save_path)

            # some experiment may not exist
            exp_id = xmldom.parse(tmp_xml_save_path).documentElement.getElementsByTagName('id')[0].firstChild.data

            # Get 3D image 
            url_grid = get_grid_url(exp_id)
            urllibreq.urlretrieve(url_grid, tmp_zip_save_path)

        except:
            print(f'Missing {gene_sym}({id}) at {timestamp} !', flush=True)
            continue

        
        # Process the .zip file
        zipfile.ZipFile(tmp_zip_save_path).extract('density.raw', tmp_cache_dir)
        img = np.fromfile(tmp_cache_dir+'density.raw', dtype=np.float32, count=-1, sep='')
        
        # Filter images that have different shape
        if len(imgs) != 0 and img.shape != imgs[0].shape:
            print(f'Different size, filter the image {gene_sym}({id}) at {timestamp}', flush=True)
            continue
        
        # Process the image, -1 means black(0) in this dataset 
        img[img == -1] = 0
        imgs.append(img)
        gene_ids.append(id)


    # Save 3D images of this timestamp
    np.save(imgs_save_path, imgs)
    np.save(genes_save_path, gene_ids)
    print(f'Gene ID of {timestamp}: {gene_ids}', flush=True)

    # Save metadata of those 3D image
    zipfile.ZipFile(tmp_zip_save_path).extract('density.mhd', tmp_cache_dir)
    shutil.move(tmp_cache_dir+'density.mhd', meta_save_path)

    # log   
    print(f'Timestamps {timestamp} has {len(imgs)} kinds of genes', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default=DATASET_DIR)
    parser.add_argument('-q', '--query-file-path', default=query_file_path)
    parser.add_argument('-t', '--num-thread', type=int, default=1)
    args = parser.parse_args()

    num_thread = max(1, min(args.num_thread, len(TIMESTAMPS)))
    dataset_dir = args.dataset_dir
    query_file_path = args.query_file_path

    print(f'Download the dataset using {num_thread} threads, saving dataset at {dataset_dir} ...', flush=True)

    # Read gene symbols and gene IDs
    df = pd.read_csv(query_file_path)
    gene_ids = df['id']
    gene_symbols = df['gene_symbol']

    print(f'Total {len(gene_symbols)} kinds of genes', flush=True)

    # Create dir if dir not exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    st = time.time()

    # Download the data of each gene at each timestamp
    pool = Pool(processes=num_thread)
    for ts in TIMESTAMPS:
        pool.apply_async(download_data, args=(ts, dataset_dir))
    pool.close()
    pool.join()

    # for ts in TIMESTAMPS:
    #     download_data(ts, dataset_dir)

    print(time.time() - st)

# Multiple: 2038.3230655193329
# Single: 2184.5832250118256
