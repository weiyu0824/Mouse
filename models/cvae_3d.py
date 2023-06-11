import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class CVAE3d(nn.Module):
    """CVAE 3D """

    def __init__(self, 
                input_shape,
                num_genes,
                num_times,
                latent_dim=128,
                gene_em_dim=32,
                time_em_dim=32,
                beta=0.001,
                device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device
        num_channel = 16
        self.encoder = nn.Sequential(
            nn.Conv3d(1, num_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(num_channel),
            nn.LeakyReLU(),
            nn.Conv3d(num_channel, num_channel * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(num_channel * 2),
            nn.LeakyReLU(),
            nn.Conv3d(num_channel * 2, num_channel * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(num_channel * 4),
            nn.LeakyReLU(),
            nn.Conv3d(num_channel * 4, num_channel * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(num_channel * 8),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_channel * 8, num_channel * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(num_channel * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(num_channel * 4, num_channel * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(num_channel * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(num_channel * 2, num_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(num_channel),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(num_channel, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        
        # Dimentions after CNN
        self.conv_embed_shape = (num_channel * 8, input_shape[0]//16, input_shape[1]//16, input_shape[2]//16) # EDIT
        conv_flat_dim = np.prod(self.conv_embed_shape)

        # Define the mean and logvar layers
        self.fc_mean = nn.Linear(conv_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat_dim, latent_dim)
        self.em_gene = nn.Embedding(num_genes, gene_em_dim)
        self.em_time = nn.Embedding(num_times, time_em_dim)
        
        # Linear layer before feeding into decoder
        self.fc_1 = nn.Linear(latent_dim+gene_em_dim+time_em_dim, conv_flat_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
    def decode(self, z, genes, times):
        genes = self.em_gene(genes)
        times = self.em_time(times)
        z = torch.cat([z, genes, times], dim=1)
        z = self.fc_1(z)
        s = self.conv_embed_shape
        z = z.view(z.shape[0], s[0], s[1], s[2], s[3])
        z = self.decoder(z)
        return z
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x, genes, times):
        x = torch.unsqueeze(x, axis=1)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z, genes, times)
        recon_x = torch.squeeze(recon_x, axis=1)
        return recon_x, mean, logvar
    
    def sample(self, genes, times):
        batch_size = genes.shape[0]
        # sample a noise from normal distribution mean=0, std=I
        z = torch.randn((batch_size, self.latent_dim)).view(batch_size, -1).to(device=self.device)

        # Add the z
        embeddding = z
        recon_x = self.decode(embeddding, genes, times)
        recon_x = torch.squeeze(recon_x, axis=1)
        return recon_x

    def loss_fn(self, recon_x, x, mean, logvar, eps=0):
        # Compute the reconstruction loss
        # recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        square_diff = (recon_x - x) ** 2
        weight = torch.where(x > 0, 1.0, eps)
        square_loss = square_diff * weight
        recon_loss = torch.mean(square_loss)

        # Compute the KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(
            1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0)
        # kl_loss = 0
        # Compute the total loss
        total_loss = recon_loss + self.beta*kl_loss

        return total_loss, recon_loss, kl_loss