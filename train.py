import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psrn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from dataset import BrainDataset
from models.CVAE import CVAE
import torchvision.transforms as transforms
import numpy as np

from torch.backends import cudnn
cudnn.benchmark = True # fast training


# Output file directories
DATA_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess/'
DATA3D_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess_3d/'

# Hyper-parameters
EPOCH = 200
BATCH_SIZE = 128

# For Model
IMG_SHAPE = (96, 144, 72)
input_shape = (IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
latent_dim = 128

# Parameters
num_genes = 2107 + 1
num_ts = 7
device = 'cuda'

# Debug 
save_slices = [10, 20, 30 ,40 ,50, 100, 500]

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.0018], std=[0.0028])
])

def init_models(ckpt_path=None):
    epoch = 1
    cvae = CVAE(input_shape, 
                num_label1=num_genes, num_label2=num_ts, 
                latent_dim=latent_dim, device=device)
    optim = torch.optim.Adam(cvae.parameters(), lr=1e-3)

    # Load existed checkpoint
    if ckpt_path != None:
        ckpt = torch.load(ckpt_path, map_location=device)
        cvae.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        epoch = ckpt.get('epoch', 1)

    return cvae, optim, epoch 

def save_slice(save_path, slice):
    plt.figure()
    plt.imshow(slice[0], cmap='tab10')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def test_step(model: CVAE, data_loader):
    # Testing
    slice = 0
    running_psnr = 0
    running_ssim = 0
    running_mse = 0
    dataset_size = len(data_loader)

    # for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Testing'):
    for imgs, time_ids, gene_ids in data_loader:
        model.eval()
        with torch.no_grad():
            # Put data to device
            imgs = imgs.to(device, dtype=torch.float)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            # slice_ids = slice_ids.to(device)

            recon_imgs = model.sample(gene_ids, time_ids).cpu().numpy() # -1 ~ 1
            imgs = imgs.cpu().numpy()

            for img, recon_img in zip (imgs, recon_imgs):

                eps = 0.0001
                # test_psnr.append(psrn(img/(max(eps, np.max(img))), recon_img/(max(eps, np.max(img)))))
                running_psnr += 1
                running_ssim += ssim(np.squeeze(img), np.squeeze(recon_img), data_range=1)
                running_mse += mse(img, recon_img)

                # slice += 1
                # if slice in save_slices:
                #     save_slice(f'result/{slice}_org.png', img)
                #     save_slice(f'result/{slice}_recon.png', recon_img)
    
    running_psnr /= dataset_size
    running_ssim /= dataset_size
    running_mse /= dataset_size

    return running_psnr, running_ssim, running_mse

def train_step(model, optim, data_loader):

    running_loss = 0
    dataset_size = len(data_loader)

    # for imgs, time_ids, gene_ids in tqdm(data_loader, desc='Training'):
    for imgs, time_ids, gene_ids in data_loader:

        imgs = imgs.to(device, dtype=torch.float)
        gene_ids = gene_ids.to(device)
        time_ids = time_ids.to(device)
        # slice_ids = slice_ids.to(device)

        # Calculate loas and gradients
        optim.zero_grad() 
        recon_x, mean, logvar = model(imgs, gene_ids, time_ids)
        loss, recon_loss, kl_loss = model.loss_fn(recon_x, imgs, mean, logvar)
        loss.backward()
        optim.step()

        running_loss += loss.cpu().data * imgs.shape[0]

    return running_loss / dataset_size

def main(data_dir):
    print(f'Preparing the dataset from {data_dir}...')

    train_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True, transform=None)
    test_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True, transform=None)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=False, num_workers=1)

    cvae, optim, start_epoch = init_models() 
    cvae = cvae.to(device) 
    
    print('Start the training process ...')
    for e in range(start_epoch, start_epoch+EPOCH):
        
        # Training
        epoch_loss = train_step(cvae, optim, train_loader)

        # Tesing
        test_psrn, test_ssim, test_mse = test_step(cvae, test_loader)
        
        template = 'Epoch {:0}, Loss: {:.4f}, Test PSRN: {:.4f}, Test SSIM: {:.4f}, Test MSE: {:.4f}'
        print(template.format(e, 
                            epoch_loss,
                            test_psrn,
                            test_ssim,
                            test_mse), flush=True)   
        torch.save({
            'epoch': e,
            'optim_state_dict': optim.state_dict(),
            'model_state_dict':  cvae.state_dict()
        }, f'ckpts/{e}_checkpoint.pth')
        
            
def plot_hist():
    data_dir = DATA_DIR
    train_set = BrainDataset(annotation_file=data_dir+'annotation.csv', data_dir=data_dir, train=True, transform=None)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    max_vals = []
    num_zero = 0
    total = 0
    clean = 0
    over = 0
    for imgs, time_ids, gene_ids, slice_ids in tqdm(train_loader, desc='Ploting'):
        imgs = imgs.numpy()

        for img in imgs:
            if total == 0:
                print(img.shape)

            max_val = np.max(img) 
            if (max_val== 0):
                num_zero += 1
            if (max_val > 0.25):
                clean += 1
                index = np.argwhere(img > 0.25)
                over += index.shape[0]
                
            else:
                max_vals.append(np.max(img)*100)

            total += 1

    plt.figure()
    plt.hist(max_vals, color='green', bins=5000)
    plt.savefig('hist.png')
    print(num_zero)
    print(total)
    print(clean)
    print(max(max_vals))
    print(over)

if __name__ == "__main__":
    main(data_dir=DATA3D_DIR)
    # plot_hist()

    # arr = np.array([[1, 2], [3, 4], [3, 5]])
    # index = np.argwhere(arr > 2)
    # print(index.shape)