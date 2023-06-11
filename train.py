import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from dataset import BrainDataset, SeqBrainDataset
from models.cvae import CVAE
from models.guide_cvae import GuideCVAE
from models.vit_ae import VITAE
from models.guide_cvae_3d import GuideCVAE3d
from models.cvae_3d import CVAE3d
from models.Sagittarius import Sagittarius
import numpy as np
from scipy.stats import spearmanr
from plot import plot_mid_slice, plot_every_slice
from preprocess import remove_pad, gamma_decode
import argparse
from dataset import load_data
from config import DATA2D_DIR, DATA3D_DIR, SHAPES, IMG_SHAPE, DEVICE, GUIDE_DIR
import copy

from torch.backends import cudnn
# cudnn.enabled = True
cudnn.benchmark = True # fast training

# Hyper-parameters
EPOCH = 100

# For Model
IMG_SHAPE = (80, 144, 96)
input_shape = (IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
latent_dim = 128
learning_rate = 5e-4 # 1e-3

# Parameters
num_genes = 2107 + 1
num_ts = 7
save_img_ids = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
post_threashold = 5e-4

#
preload_guides = []
for i in range(7):
    preload_guides.append(np.load(f'{GUIDE_DIR}time_{i}.npy'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch_matric(imgs, recon_imgs, time_ids):
    """
    :param imgs: batch of images 
    :param recon_imgs: batch of reconstructed images
    """

    test_psnr = []
    test_ssim = []
    test_mse = []
    test_spearmanr = []
    ndata = imgs.shape[0]

    for img, recon_img, time_id in zip (imgs, recon_imgs, time_ids):
        # Remove padding & remove gamma encoded
        org_shape = SHAPES[time_id]
        img = gamma_decode(remove_pad(img, org_shape))
        
        recon_img = gamma_decode(remove_pad(recon_img, org_shape))

        # Post process
        # recon_img = np.where(recon_img > post_threashold, recon_img, 0)

        # Adrress nan problem
        if np.max(recon_img) == 0:
            print('Recon Image is empty', np.max(p))
            test_psnr.append(0)
            test_ssim.append(0)
            # test_mse
            test_spearmanr.append(0)
            continue

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
        coef = spearmanr(img.flatten(), recon_img.flatten())[0]
        test_spearmanr.append(coef)
        
        # Debug
        if np.max(img) == 0:
            print('Image is empty')
            exit()

    return sum(test_psnr)/ndata, sum(test_ssim)/ndata, sum(test_mse)/ndata, sum(test_spearmanr)/ndata

def init_models(model_type, device, ckpt_path=None):
    """
    param: model_type: 'cvae', 'guide_cvae'
    """
    epoch = 1
    if model_type == 'cvae':
        model = CVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts, 
                latent_dim=latent_dim, device=device)
    elif model_type == 'cvae_3d':
        model = CVAE3d(input_shape, num_genes, num_ts, 
                latent_dim=latent_dim, device=device)
    elif model_type == 'guide_cvae':
        model = GuideCVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts,
                latent_dim=latent_dim, device=device)
    elif model_type == 'guide_cvae_3d':
        model = GuideCVAE3d(input_shape, num_genes, num_ts, 
                latent_dim=latent_dim, device=device)
    elif model_type == 'sag':
        model = Sagittarius(
            input_dim=IMG_SHAPE, num_classes=1, class_sizes=[num_genes],
            latent_dim=128, cvae_yemb_dims=[128], cvae_hidden_dims=[128], 
            temporal_dim=1, cat_dims=[128], num_heads=8, num_ref_points=3, 
            minT=0, maxT=6, device=device)
    else:
        raise ValueError('Unsupport model type')
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(count_parameters(model))
    # Load existed checkpoint
    if ckpt_path != None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        epoch = ckpt.get('epoch', 1)
    
    return model, optim, epoch 

def prepare_dataset(model_type, data_dir, batch_size=1):
    print(f'Preparing the dataset from {data_dir}...')

    if model_type in [ 'cvae', 'guide_cvae', 'cvae_3d', 'guide_cvae_3d']:
        train_set = BrainDataset(annotation_file=data_dir+'train_annotation.csv', data_dir=data_dir, transform=None)
        test_set = BrainDataset(annotation_file=data_dir+'test_annotation.csv', data_dir=data_dir, transform=None)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, num_workers=1)
    elif model_type in ['sag']:
        data = load_data(DATA3D_DIR, 'train_annotation.csv', 'test_annotation.csv')
        train_set = SeqBrainDataset(data['train_annot'], data['train_imgs'], train=True)
        test_set = SeqBrainDataset(data['train_annot'], data['train_imgs'], data['test_annot'], data['test_imgs'], train=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=1)
    return train_loader, test_loader

def test_step(model, data_loader, model_type, device='cpu', save_img_ids=[], save_dir='./'):
    # Testing
    test_psnr = []
    test_ssim = []
    test_mse = []
    test_spearnam = []

    # 
    img_id = 0

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Testing'):
    # for imgs, time_ids, gene_ids in data_loader:
        model.eval()
        with torch.set_grad_enabled(False):
            imgs = imgs.to(device, dtype=torch.float)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            guides = torch.tensor(np.array([preload_guides[time_id] for time_id in time_ids]))
            guides = guides.to(device, dtype=torch.float)

            if model_type in ['cvae', 'cvae_3d']:
                recon_imgs = model.sample(gene_ids, time_ids).cpu().numpy()
            elif model_type in ['guide_cvae', 'guide_cvae_3d']:
                recon_imgs = model.sample(gene_ids, time_ids, guides).cpu().numpy()
            else:
                raise ValueError('Unimplement test step error')

            # Remove from gpu
            # imgs.detach(), gene_ids.detach(), time_ids.detach(), guides.detach(), guides.detach()

            imgs = imgs.cpu().numpy()
            time_ids = time_ids.cpu().data

            batch_psnr, batch_ssim, batch_mse, batch_spearmanr = batch_matric(imgs, recon_imgs, time_ids)
            test_psnr.append(batch_psnr)
            test_ssim.append(batch_ssim)
            test_mse.append(batch_mse)
            test_spearnam.append(batch_spearmanr)

            # Save image if specified
            if len(save_img_ids) != 0:
                if img_id in save_img_ids:
                    img = imgs[0]
                    time_id = time_ids[0]
                    recon_img = recon_imgs[0]
                    org_shape = SHAPES[time_id]
                    img = gamma_decode(remove_pad(img, org_shape))
                    recon_img = gamma_decode(remove_pad(recon_img, org_shape))
                    recon_img = np.where(recon_img > post_threashold, recon_img, 0)
                    
                    plot_every_slice(img, save_path=f'{save_dir}org_{img_id}.png')
                    plot_every_slice(recon_img, save_path=f'{save_dir}recon_{img_id}.png')
                img_id += 1

            
    return sum(test_psnr)/len(test_psnr), sum(test_ssim)/len(test_ssim), sum(test_mse)/len(test_mse), sum(test_spearnam)/len(test_spearnam)

def test_step_seq(model, data_loader, model_type, device='cpu', save_img_ids=[], save_dir='./'):
    test_psnr = []
    test_mse = []
    test_ssim = []
    test_spearnam = []

    # 
    img_id = 0

    for img_series, time_series, masks, genes in tqdm(data_loader, desc='test-seq'):
        img_series = img_series.to(device, dtype=torch.float)
        time_series = time_series.to(device)
        masks = masks.to(device)
        genes = genes.to(device)
        ys = [genes]   

        model.eval()
        with torch.set_grad_enabled(False):
            recon_img_series, mean, logvar = model(img_series, time_series, ys, masks)
            loss_dict = model.loss_fn(img_series, recon_img_series, mean, logvar)
            # print(loss_dict['loss'], loss_dict['MSE'], loss_dict['KLD'])

            img_series = img_series.cpu().numpy()
            time_series = time_series.cpu().data
            recon_img_series = recon_img_series.cpu().numpy()
            masks = masks.cpu().numpy()
            
            #
            pred_indices = np.argwhere(masks)
            imgs = []
            recon_imgs = []
            time_ids = []
            for id, pos in zip(pred_indices[:, 0], pred_indices[:, 1]):
                imgs.append(img_series[id][pos])
                recon_imgs.append(recon_img_series[id][pos])
                time_ids.append(time_series[id][pos])

            imgs = np.array(imgs)
            recon_imgs = np.array(recon_imgs)
            time_ids = np.array(time_ids)

            # Save image if specified
            if len(save_img_ids) != 0:
                if img_id in save_img_ids:
                    img = imgs[0]
                    time_id = time_ids[0]
                    recon_img = recon_imgs[0]
                    org_shape = SHAPES[time_id]
                    img = gamma_decode(remove_pad(img, org_shape))
                    recon_img = gamma_decode(remove_pad(recon_img, org_shape))
                    recon_img = np.where(recon_img > post_threashold, recon_img, 0)
                    
                    plot_every_slice(img, save_path=f'{save_dir}org_{img_id}.png')
                    plot_every_slice(recon_img, save_path=f'{save_dir}recon_{img_id}.png')
                img_id += 1

            batch_psnr, batch_ssim, batch_mse, batch_spearmanr = batch_matric(imgs, recon_imgs, time_ids)
            test_psnr.append(batch_psnr)
            test_ssim.append(batch_ssim)
            test_mse.append(batch_mse)
            test_spearnam.append(batch_spearmanr)
    
    return sum(test_psnr)/len(test_psnr), sum(test_ssim)/len(test_ssim), sum(test_mse)/len(test_mse), sum(test_spearnam)/len(test_spearnam)

def train_step(model, optim, data_loader, model_type, device='cpu'):

    epoch_loss = []
    epoch_recon_loss = []

    for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Training'):
        imgs = imgs.to(device, dtype=torch.float)
        gene_ids = gene_ids.to(device)
        time_ids = time_ids.to(device)
        guides = torch.tensor(np.array([preload_guides[time_id] for time_id in time_ids]))
        guides = guides.to(device, dtype=torch.float)

        optim.zero_grad() 
        with torch.set_grad_enabled(True):
            if model_type in ['cvae', 'cvae_3d']:
                recon_x, mean, logvar = model(imgs, gene_ids, time_ids)
            elif model_type in ['guide_cvae', 'guide_cvae_3d']:
                recon_x, mean, logvar = model(imgs, gene_ids, time_ids, guides)
            else:
                raise ValueError('Unimplement train step error')

            loss, recon_loss, kl_loss = model.loss_fn(recon_x, imgs, mean, logvar)
            # print(loss, recon_loss, kl_loss)
            loss.backward()
            optim.step()
        
        # Remove from gpu
        # imgs.detach(), gene_ids.detach(), time_ids.detach(), guides.detach(), guides.detach(), recon_x.detach()


        epoch_loss.append(loss.cpu().data)
        epoch_recon_loss.append(recon_loss.cpu().data)

    return sum(epoch_loss)/len(epoch_loss), sum(epoch_recon_loss)/len(epoch_recon_loss)

def train_step_seq(model, optim, data_loader, model_type, device='cpu'):
    epoch_loss = []
    epoch_recon_loss = []

    for img_series, time_series, masks, genes in tqdm(data_loader, desc='train-seq'):
        img_series = img_series.to(device, dtype=torch.float)
        time_series = time_series.to(device)
        masks = masks.to(device)
        genes = genes.to(device)
        ys = [genes]

        optim.zero_grad() 
        with torch.set_grad_enabled(True):
            recon_img_series, mean, logvar = model(img_series, time_series, ys, masks)
            loss_dict = model.loss_fn(img_series, recon_img_series, mean, logvar)
            # print(loss_dict['loss'], loss_dict['MSE'], loss_dict['KLD'])
            loss_dict['loss'].backward()
            optim.step()
            epoch_loss.append(loss_dict['loss'].cpu().data)
            epoch_recon_loss.append(loss_dict['MSE'].cpu().data)

    return sum(epoch_loss)/len(epoch_loss), sum(epoch_recon_loss)/len(epoch_recon_loss)
    # return 0, 0

def train(data_dir, model_type, ckpt_path, batch_size, device='cpu'):

    train_loader, test_loader = prepare_dataset(model_type, data_dir, batch_size=batch_size)
    model, optim, start_epoch = init_models(model_type, device, ckpt_path)
    
    print('Start the training process ...')
    for e in range(start_epoch, start_epoch+EPOCH):
        
        if model_type in [ 'cvae', 'guide_cvae', 'cvae_3d', 'guide_cvae_3d']:
            epoch_loss, train_recon_loss = train_step(model, optim, train_loader, model_type, device=device)
            test_psrn, test_ssim, test_mse, test_spearmanr = test_step(model, test_loader, model_type, device=device)
        elif model_type in ['sag']:
            
            # epoch_loss, train_recon_loss = 0, 0
            epoch_loss, train_recon_loss = train_step_seq(model, optim, train_loader, model_type, device=device)
            test_psrn, test_ssim, test_mse, test_spearmanr = test_step_seq(model, test_loader, model_type, device=device)
            # exit()

        template = 'Epoch {:0}, Loss: {:.6f}, Train MSE: {:.6f}, Test PSNR: {:.6f}, Test SSIM: {:.6f}, Test MSE: {:.6f}, Test Spearman: {:6f}'
        print(template.format(e, 
                            epoch_loss,
                            train_recon_loss,
                            test_psrn,
                            test_ssim,
                            test_mse, 
                            test_spearmanr), flush=True)
        torch.save({
            'epoch': e,
            'optim_state_dict': optim.state_dict(),
            'model_state_dict':  model.state_dict()
        }, f'ckpts/{model_type}_{e}_checkpoint.pth')

def evaluate(data_dir, model_type, ckpt_path, save_dir, device='cpu'):
    # test_set = BrainDataset(annotation_file=data_dir+'test_annotation.csv', data_dir=data_dir, transform=None)
    # test_loader = DataLoader(test_set, batch_size=1, pin_memory=False, num_workers=1)

    _, test_loader = prepare_dataset(model_type, data_dir, batch_size=1)

    model, _, _ = init_models(model_type, device, ckpt_path)
    # model = None
    if model_type in [ 'cvae', 'guide_cvae', 'cvae_3d', 'guide_cvae_3d']:
        test_psnr, test_ssim, test_mse, test_spearmanr = test_step(model, test_loader, model_type, 
                                                               save_img_ids=save_img_ids, save_dir=save_dir, device=device)
    elif model_type in ['sag']:
        test_psnr, test_ssim, test_mse, test_spearmanr = test_step_seq(model, test_loader, model_type, 
                                                               save_img_ids=save_img_ids, save_dir=save_dir, device=device)

    template = 'Test PSNR: {:.6f}, Test SSIM: {:.6f}, Test MSE: {:.6f}, Test Spearman: {:6f}'
    print(template.format(test_psnr, 
                          test_ssim,
                          test_mse, 
                          test_spearmanr), flush=True)   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('model_type', choices=['cvae', 'cvae_3d', 'guide_cvae', 'guide_cvae_3d', 'sag', 'guide'])
    parser.add_argument('-ckpt_path',  default=None)
    parser.add_argument('-data_dir',  default=DATA3D_DIR) 
    parser.add_argument('-save_dir',  default='.')
    parser.add_argument("-batch_size", default=32, type=int, help="batch size")
    parser.add_argument('-device', default=DEVICE)
    
    args = parser.parse_args()
    command = args.command

    model_type = args.model_type
    ckpt_path = args.ckpt_path
    data_dir = args.data_dir
    save_dir = args.save_dir
    batch_size = args.batch_size
    device = args.device

    print(f'model_type={model_type}, ckpt_path={ckpt_path}, data_dir={data_dir}, save_dir={save_dir}, device={device}, batch_size={batch_size}')

    if command == 'train':
        train(data_dir, model_type, ckpt_path, batch_size=batch_size, device=device)
    elif command == 'eval':
        evaluate(data_dir, model_type, ckpt_path, save_dir=save_dir, device=device)

if __name__ == "__main__":
    main()