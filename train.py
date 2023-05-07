import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from dataset import BrainDataset
from models.CVAE import CVAE
from models.GuideCVAE import GuideCVAE
import torchvision.transforms as transforms
import numpy as np
from plot import plot_mid_slice, plot_every_slice
from preprocess import remove_pad, gamma_decode

from torch.backends import cudnn
cudnn.benchmark = True # fast training


# Output file directories
DATA_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess/'
DATA3D_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess_3d/'
SHAPES = [(40, 75, 70), (69, 109, 89), (65, 132, 94), (40, 43, 67), (50, 43, 77), (50, 40, 68), (58, 41, 67)]


# Hyper-parameters
EPOCH = 10
BATCH_SIZE = 128

# For Model
IMG_SHAPE = (80, 144, 96)
input_shape = (IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
latent_dim = 128
learning_rate = 1e-3 # 1e-3

# Parameters
num_genes = 2107 + 1
num_ts = 7
device = 'cuda:3'

# Debug 
save_slices = [10, 200, 3000 ,4000 ,5000, 100, 500]

# transform = transforms.Compose([
#     # transforms.ToTensor(),
#     transforms.Normalize(mean=[0.0018], std=[0.0028])
# ])

def init_models(ckpt_path=None):
    epoch = 1
    cvae = CVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts, 
                latent_dim=latent_dim, device=device)
    optim = torch.optim.Adam(cvae.parameters(), lr=learning_rate)

    # Load existed checkpoint
    if ckpt_path != None:
        ckpt = torch.load(ckpt_path, map_location=device)
        cvae.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        epoch = ckpt.get('epoch', 1)

    return cvae, optim, epoch 

def test_step(model: CVAE, data_loader, preload_guides):
    # Testing
    slice = 0
    test_psnr = []
    test_ssim = []
    test_mse = []

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Testing'):
    # for imgs, time_ids, gene_ids in data_loader:
        model.eval()
        with torch.no_grad():
            # Put data to device
            imgs = imgs.to(device, dtype=torch.float) # shape=(batch, channel, height, width)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            # slice_ids = slice_ids.to(device)

            recon_imgs = model.sample(gene_ids, time_ids).cpu().numpy() # -1 ~ 1
            imgs = imgs.cpu().numpy()
            time_ids = time_ids.cpu().data

            for img, recon_img, time_id in zip (imgs, recon_imgs, time_ids):
                # 
                org_shape = SHAPES[time_id]
                
                # img = remove_pad(img, org_shape)
                # recon_img = remove_pad(recon_img, org_shape)

                ssim_per_3d = 0
                psnr_per_3d = 0
                num_nonzero_slice = 0
                for i in range(img.shape[0]):
                    ssim_per_3d += ssim(img[i], recon_img[i], data_range=1)
                    psnr_per_3d += psnr(img[i], recon_img[i], data_range=1)
                    num_nonzero_slice += 1

                ssim_per_3d /= num_nonzero_slice
                psnr_per_3d /= num_nonzero_slice

                test_psnr.append(psnr_per_3d)
                test_ssim.append(ssim_per_3d)
                test_mse.append(mse(img, recon_img))


                if slice in save_slices:
                    show_mid_slice(img, save_path=f'result/{slice}_org.png')
                    show_mid_slice(recon_img, save_path=f'result/{slice}_recon.png')
                slice += 1
                # # Debug
                # if ssim_per_3d > 0.7:
                #     show_mid_slice(img, save_path=f'result/{slice}_org.png')
                #     show_mid_slice(recon_img, save_path=f'result/{slice}_recon.png')
                #     print(np.max(img))
                #     print(np.max(recon_img))
                #     print('---')
                #     slice += 1
    return sum(test_psnr)/len(test_psnr), sum(test_ssim)/len(test_ssim), sum(test_mse)/len(test_mse)

def test_step_guide(model: GuideCVAE, data_loader, preload_guides):
    # Testing
    slice = 0
    test_psnr = []
    test_ssim = []
    test_mse = []

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Testing'):
    # for imgs, time_ids, gene_ids in data_loader:
        model.eval()
        with torch.no_grad():
            # Put data to device
            imgs = imgs.to(device, dtype=torch.float) # shape=(batch, channel, height, width)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            
            guides = torch.tensor(np.array([preload_guides[time_id] for time_id in time_ids]))
            guides = guides.to(device, dtype=torch.float)
            # print(guides.shape)

            recon_imgs = model.sample(gene_ids, time_ids, guides).cpu().numpy() # -1 ~ 1
            imgs = imgs.cpu().numpy()
            time_ids = time_ids.cpu().data

            for img, recon_img, time_id in zip (imgs, recon_imgs, time_ids):
                # Remove padding & remove gamma encoded
                org_shape = SHAPES[time_id]
                img = gamma_decode(remove_pad(img, org_shape))
                recon_img = gamma_decode(remove_pad(recon_img, org_shape))

                ssim_per_3d = 0
                psnr_per_3d = 0
                num_nonzero_slice = 0
                for i in range(img.shape[0]):
                    ssim_per_3d += ssim(img[i], recon_img[i], data_range=1)
                    psnr_per_3d += psnr(img[i], recon_img[i], data_range=1)
                    num_nonzero_slice += 1

                ssim_per_3d /= num_nonzero_slice
                psnr_per_3d /= num_nonzero_slice

                test_psnr.append(psnr_per_3d)
                test_ssim.append(ssim_per_3d)
                test_mse.append(mse(img, recon_img))


                if slice in save_slices:
                    plot_every_slice(img, save_path=f'result/guide/{slice}_org.png')
                    plot_every_slice(recon_img, save_path=f'result/guide/{slice}_recon.png')
                slice += 1
    return sum(test_psnr)/len(test_psnr), sum(test_ssim)/len(test_ssim), sum(test_mse)/len(test_mse)

def train_step_guide(model: GuideCVAE, optim, data_loader, preload_guides):

    running_loss = 0
    running_recon_loss = 0
    dataset_size = len(data_loader)

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Training'):
    # for imgs, time_ids, gene_ids in data_loader:

        imgs = imgs.to(device, dtype=torch.float)
        gene_ids = gene_ids.to(device)
        time_ids = time_ids.to(device)
        guides = torch.tensor(np.array([preload_guides[time_id] for time_id in time_ids]))
        guides = guides.to(device, dtype=torch.float)
        # print(guides.shape)

        # Calculate loas and gradients
        optim.zero_grad() 
        recon_x, mean, logvar = model(imgs, gene_ids, time_ids, guides)
        loss, recon_loss, kl_loss = model.loss_fn(recon_x, imgs, mean, logvar)
        print(recon_loss*10000, kl_loss)
        loss.backward()
        optim.step()
        running_loss += loss.cpu().data 
        running_recon_loss += recon_loss.cpu().data

    return running_loss / dataset_size, running_recon_loss / dataset_size


def train_step(model, optim, data_loader):

    running_loss = 0
    running_recon_loss = 0
    dataset_size = len(data_loader)

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Training'):
    # for imgs, time_ids, gene_ids in data_loader:

        imgs = imgs.to(device, dtype=torch.float)
        gene_ids = gene_ids.to(device)
        time_ids = time_ids.to(device)
        # slice_ids = slice_ids.to(device)

        # Calculate loas and gradients
        optim.zero_grad() 
        recon_x, mean, logvar = model(imgs, gene_ids, time_ids)
        loss, recon_loss, kl_loss = model.loss_fn(recon_x, imgs, mean, logvar)
        print(recon_loss*10000, kl_loss)
        loss.backward()
        optim.step()
        running_loss += loss.cpu().data 
        running_recon_loss += recon_loss.cpu().data

    return running_loss / dataset_size, running_recon_loss / dataset_size

def main(data_dir):
    print(f'Preparing the dataset from {data_dir}...')

    train_set = BrainDataset(annotation_file=data_dir+'train_annotation.csv', data_dir=data_dir, transform=None)
    test_set = BrainDataset(annotation_file=data_dir+'test_annotation.csv', data_dir=data_dir, transform=None)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=False, num_workers=1)

    # cvae, optim, start_epoch = init_models() 
    # cvae = cvae.to(device) 
    
    # print('Start the training process ...')
    # for e in range(start_epoch, start_epoch+EPOCH):
        
    #     # Training
    #     epoch_loss, train_recon_loss = train_step(cvae, optim, train_loader)

    #     # Tesing
    #     test_psrn, test_ssim, test_mse = test_step(cvae, test_loader)
        
    #     template = 'Epoch {:0}, Loss: {:.6f}, Train MSE: {:.6f}, Test PSNR: {:.6f}, Test SSIM: {:.6f}, Test MSE: {:.6f}'
    #     print(template.format(e, 
    #                         epoch_loss,
    #                         train_recon_loss,
    #                         test_psrn,
    #                         test_ssim,
    #                         test_mse), flush=True)   
        
    #     torch.save({
    #         'epoch': e,
    #         'optim_state_dict': optim.state_dict(),
    #         'model_state_dict':  cvae.state_dict()
    #     }, f'ckpts/{e}_checkpoint.pth')


    preload_guides = []
    for i in range(7):
        preload_guides.append(np.load(f'/m-ent1/ent1/wylin6/mouse/guide/time_{i}.npy'))
        preload_guides[i] = preload_guides[i]
    
    start_epoch = 0
    model = GuideCVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts, 
                latent_dim=latent_dim, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device) 
    
    print('Start the training process ...')
    for e in range(start_epoch, start_epoch+EPOCH):
        
        # Training
        for _ in range(2):
            epoch_loss, train_recon_loss = train_step_guide(model, optim, train_loader, preload_guides)

        # Tesing
        test_psrn, test_ssim, test_mse = test_step_guide(model, test_loader, preload_guides)
        
        template = 'Epoch {:0}, Loss: {:.6f}, Train MSE: {:.6f}, Test PSNR: {:.6f}, Test SSIM: {:.6f}, Test MSE: {:.6f}'
        print(template.format(e, 
                            epoch_loss,
                            train_recon_loss,
                            test_psrn,
                            test_ssim,
                            test_mse), flush=True)   
        
        torch.save({
            'epoch': e,
            'optim_state_dict': optim.state_dict(),
            'model_state_dict':  model.state_dict()
        }, f'ckpts/{e}_checkpoint.pth')
    

                

if __name__ == "__main__":
    main(data_dir=DATA3D_DIR)
    # plot_hist()

    # arr = np.array([[1, 2], [3, 4], [3, 5]])
    # index = np.argwhere(arr > 2)
    # print(index.shape)