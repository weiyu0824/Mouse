import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psrn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from dataset import BrainDataset
from models.CVAE import CVAE

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
device = 'cuda:0'


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
        losses = []

        # Training
        for imgs, time_ids, gene_ids, slice_ids in tqdm.tqdm(train_loader):

            # Put data to device
            imgs = imgs.to(device)
            gene_ids = gene_ids.to(device)
            time_ids = time_ids.to(device)
            slice_ids = slice_ids.to(device)

            # Calculate loas and gradients
            optim.zero_grad() 
            recon_x, mean, logvar = cvae(imgs, gene_ids, time_ids, slice_ids)
            loss = cvae.loss_fn(recon_x, imgs, mean, logvar)
            losses.append(loss)
            loss.backward()
            optim.step()
        
        # Testing
        for imgs, time_ids, gene_ids, slice_ids in tqdm.tqdm(test_loader):
            cvae.eval()
            with torch.no_grad():
                recon_imgs = cvae.sample()

                psrn_loss = psrn(imgs, recon_imgs)
                ssim_loss = ssim(imgs, recon_imgs)
                mse_loss = mse(imgs, recon_imgs)

        
        template = 'Epoch {:0}, Loss: {:.4f}, Test PSRN Loss: {:.4f}, Test SSIM Loss: {:.4f}, Test MSE Loss: {:.4f}'
        print(template.format(e, 
                              sum(losses)/len(losses),
                              psrn_loss,
                              ssim_loss,
                              mse_loss))   
        
            
        


if __name__ == "__main__":
    main()
