import numpy as np
from config import TIMESTAMPS, SHAPES, DATASET_DIR, DATA2D_DIR, DATA3D_DIR, GUIDE_DIR, IMG_SHAPE
from plot import plot_every_slice


ss = {}
ss[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ss[1] = [0, 1, 2, 5, 7]
ss[2] = [0, 1, 2, 5, 7]
ss[3] = [0, 1, 2, 5, 7]
# ss[4] = [0, 1, 2, 5, 7]
# ss[5] = [0, 1, 2, 5, 7]
ss[4] = [0, 1, 2]
ss[5] = [0, 1, 2]
ss[6] = [0, 1, 2, 7]


# struct that we are going to test
target_timestamp = [4, 5]

gene_2_imgs = {}


dataset_dir = DATASET_DIR

import tqdm
def cal_first_rank_ratio():
    """
    Ratio that can retain 1st ranking
    """
    for target_time in target_timestamp:
        shape = SHAPES[target_time]
        ts = TIMESTAMPS[target_time]

        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')

        for img_id, gene_id in enumerate(gene_ids):
            img_3d = np.reshape(imgs[img_id], shape)

            if gene_id not in gene_2_imgs:
                gene_2_imgs[gene_id] = []
            gene_2_imgs[gene_id].append(img_3d)

    t1_struct = np.load(f'ontology_{TIMESTAMPS[target_timestamp[0]]}.npy')
    t2_struct = np.load(f'ontology_{TIMESTAMPS[target_timestamp[1]]}.npy')
    t1_struct = np.reshape(t1_struct, -1)
    t2_struct = np.reshape(t2_struct, -1)

    t1_champs = []
    t2_champs = []

    def cal_dominate_struct(score):
            max_avg_score = 0
            sid_with_max_score = 0
            for score_key in score.keys():
                avg_score = sum(score[score_key]) / len(score[score_key])
                if avg_score > max_avg_score:
                    sid_with_max_score = score_key
            return sid_with_max_score

    for gene_key in tqdm.tqdm(gene_2_imgs.keys()):
        if len(gene_2_imgs[gene_key]) != 2:
            continue

        img1 = np.reshape(gene_2_imgs[gene_key][0], -1)
        img2 = np.reshape(gene_2_imgs[gene_key][1], -1)

        score = {}
        for sval, im_val in zip(t1_struct, img1):
            if sval == 0:
                continue
            # print(sval, ss[target_timestamp[0]])
            if sval not in ss[target_timestamp[0]]: # filter out struct that we dont care about
                continue
            
            if sval not in score:
                score[sval] = []
            score[sval].append(im_val)
        t1_champs.append(cal_dominate_struct(score))

        score = {}
        for sval, im_val in zip(t2_struct, img2):
            if sval not in ss[target_timestamp[1]]: # filter out struct that we dont care about
                continue
            if sval not in score:
                score[sval] = []
            score[sval].append(im_val)
        t2_champs.append(cal_dominate_struct(score))
    
    retain_num = 0
    for t1_champ, t2_champ in zip(t1_champs, t2_champs):
        if t1_champ == t2_champ:
            retain_num += 1

    print('target time', target_timestamp)
    print(retain_num / len(t1_champs))
    print(t1_champs[:50])
    print(t2_champs[:50])


        
cal_first_rank_ratio()





def plot_slice_in_every_time():
    for tid in range(7):
        shape = SHAPES[tid]
        ts = TIMESTAMPS[tid]

        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')

        for img_id, gene_id in enumerate(gene_ids):
            img_3d = np.reshape(imgs[img_id], shape)

            if gene_id not in gene_2_imgs:
                gene_2_imgs[gene_id] = []
            gene_2_imgs[gene_id].append(img_3d)

    for gid, gene_key in enumerate(gene_2_imgs.keys()):
        id = 0
        for im in gene_2_imgs[gene_key]:
            plot_every_slice(im, save_path=f'analysis_im/{gene_key}_ss_{id}.png')
            id += 1
        
        if gid == 10: 
            break


def cal_corr_between_time():
    for target_time in target_timestamp:
        shape = SHAPES[target_time]
        ts = TIMESTAMPS[target_time]

        imgs = np.load(dataset_dir + ts + '.npy')
        gene_ids = np.load(dataset_dir + ts + '.genes.npy')

        for img_id, gene_id in enumerate(gene_ids):
            img_3d = np.reshape(imgs[img_id], shape)

            if gene_id not in gene_2_imgs:
                gene_2_imgs[gene_id] = []
            gene_2_imgs[gene_id].append(img_3d)

    for s1_val in ss[target_timestamp[0]]:
        for s2_val in ss[target_timestamp[1]]:

            print(f'EXP: {s1_val} & {s2_val}')
            target_struct = [s1_val, s2_val]

            struct_id = []
            for i in [0, 1]:
                ontology = np.load(f'ontology_{TIMESTAMPS[target_timestamp[i]]}.npy')
                struct_id.append(np.argwhere(ontology == target_struct[i]))

            genes_amount = [[], []]

            for gene_key in gene_2_imgs.keys():
                if len(gene_2_imgs[gene_key]) != 2:
                    continue
                
                imgs = gene_2_imgs[gene_key]
                for i in [0, 1]:
                    amount = 0
                    for j in struct_id[i]:
                        amount += imgs[i][j[0], j[1], j[2]]
                    genes_amount[i].append(amount)

            print(" correlation:", np.corrcoef(genes_amount[0], genes_amount[1])[0, 1])
            print('')


