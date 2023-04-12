import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psrn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from dataset import BrainDataset
from models.CVAE import CVAE
import time
import numpy as np

from torch.backends import cudnn
cudnn.benchmark = True # fast training


# Output file directories
DATASET_DIR = '/data/wylin6/mouse/dataset/'

# Hyper-parameters
EPOCH = 1000
BATCH_SIZE = 64

# For Model
IMG_SHAPE = (60, 96, 80)
input_shape = (1, IMG_SHAPE[1], IMG_SHAPE[2])
latent_dim = 128

# Parameters
num_genes = 2107 + 1
num_ts = 7
device = 'cuda'


def save_slice(save_path, slice):
    plt.figure()
    plt.imshow(slice, cmap='tab10')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def test_step(model: CVAE, test_data_loader, epoch):
    i = 0
    for img, gene, ts in test_data_loader:
        # print(gene, ts) # shape = (1, input_size), 

        # Put data to device
        gene = gene.to(device)
        ts = ts.to(device)


        reconn_img = model.sample(gene, ts) 

        # back to cpu
        reconn_img = reconn_img.cpu().detach().numpy()
        img = img.cpu().numpy()

        # save 30-th slice
        gene = gene.cpu().detach().item()
        ts = ts.cpu().detach().item()
        # save_slice(f'result/ground_{gene}_{ts}_{30}.png', img[0][30])
        save_slice(f'result/sample_{gene}_{ts}_{30}_{epoch}.png', reconn_img[0][30])

        # print(i)
        i += 1
        if i > 5:
            # exit()
            return



def main():
    print('Preparing the dataset ...')
    data_dir = '/data/wylin6/mouse/preprocess/'


    train_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True)
    test_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=24)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=24)

    # Model 
    cvae = CVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts, num_label3=IMG_SHAPE[0],
                latent_dim=latent_dim, device=device)
    optim = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    cvae = cvae.to(device)    
    
    print('Start the training process ...')
    for e in range(1, EPOCH+1):
        
        # take_speeds = []
        # put_speeds = []
        # train_speeds = []

        # Training
        # st = time.time()
        losses = []
        recon_losses = []
        kl_losses = []
        for imgs, time_ids, gene_ids, slice_ids in tqdm.tqdm(train_loader, desc='Training'):
        # for imgs, time_ids, gene_ids, slice_ids in train_loader:
            # take_speeds.append(time.time() - st)

            # st = time.time()
            # Put data to device
            imgs = imgs.to(device)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            slice_ids = slice_ids.to(device)
            # put_speeds.append(time.time() - st)

            # st = time.time()
            # Calculate loas and gradients
            optim.zero_grad() 
            recon_x, mean, logvar = cvae(imgs, gene_ids, time_ids, slice_ids)
            loss, recon_loss, kl_loss = cvae.loss_fn(recon_x, imgs, mean, logvar)
            losses.append(loss.cpu().data)
            recon_losses.append(recon_loss.cpu().data)
            kl_losses.append(kl_loss.cpu().data)
            loss.backward()
            optim.step()
            # train_speeds.append(time.time() - st)
        print(sum(recon_losses)/len(recon_losses))
        print(sum(kl_losses)/len(kl_losses))
        #     st = time.time()
        # print('------speed-------')
        # print(sum(take_speeds) / len(take_speeds))
        # print(sum(put_speeds) / len(put_speeds))
        # print(sum(train_speeds) / len(train_speeds))

        # Testing
        test_psnr_losses = []
        test_ssim_losses = []
        test_mse_lossess = []
        for imgs, time_ids, gene_ids, slice_ids in tqdm.tqdm(test_loader, desc='Testing'):
            cvae.eval()
            with torch.no_grad():
                # Put data to device

                gene_ids = gene_ids.to(device)
                time_ids = time_ids.to(device)
                slice_ids = slice_ids.to(device)

                recon_imgs = cvae.sample(gene_ids, time_ids, slice_ids).cpu().numpy()
                imgs = imgs.numpy()

                # print(recon_imgs.shape)
                # print(imgs.shape)

                test_psnr_losses.append(psrn(imgs, recon_imgs))
                test_ssim_losses.append(ssim(np.squeeze(imgs), np.squeeze(recon_imgs), data_range=1))
                test_mse_lossess.append(mse(imgs, recon_imgs))
        
        template = 'Epoch {:0}, Loss: {:.4f}, Test PSRN Loss: {:.4f}, Test SSIM Loss: {:.4f}, Test MSE Loss: {:.4f}'
        print(template.format(e, 
                              sum(losses)/len(losses),
                              sum(test_psnr_losses)/len(test_psnr_losses),
                              sum(test_ssim_losses)/len(test_ssim_losses),
                              sum(test_mse_lossess)/len(test_mse_lossess)))   
        
            
        


if __name__ == "__main__":
    main()
