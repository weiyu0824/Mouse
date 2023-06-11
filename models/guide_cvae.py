import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class GuideCVAE(nn.Module):
    """Guide CVAE"""

    def __init__(self, 
                input_shape, 
                num_label1, 
                num_label2, 
                latent_dim=128,
                label_embed_dim=32,
                beta=0.001,
                device='cpu'):
        super().__init__()

        self.show_summary = False # Edit

        self.latent_dim = latent_dim
        self.label_embed_dim = label_embed_dim
        self.beta = beta
        self.device = device

        stride = 2
        padding = 1
        
        # Dimentions after CNN
        self.conv_embed_shape = (256, input_shape[1]//16, input_shape[2]//16) # EDIT
        conv_flat_dim = self.conv_embed_shape[0] * self.conv_embed_shape[1] * self.conv_embed_shape[2]

        # Define the encoder layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=stride, padding=padding)
        self.fc1 = nn.Linear(conv_flat_dim, 256) 

        # Define the label embedding layers
        self.em_label1 = nn.Embedding(num_label1, label_embed_dim)
        self.em_label2 = nn.Embedding(num_label2, label_embed_dim)

        # label_embed_dims = 3 * label_embed_dim
        label_embed_dims = 2 * label_embed_dim

        # Define the mean and logvar layers
        self.fc_mean = nn.Linear(256 + label_embed_dims, latent_dim)
        self.fc_logvar = nn.Linear(256 + label_embed_dims, latent_dim)

        
        # Define the guided encoder layers
        self.guide_conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=stride, padding=padding)
        self.guide_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=stride, padding=padding)
        self.guide_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding)
        self.guide_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=stride, padding=padding)
        self.fc_guide = nn.Linear(conv_flat_dim, latent_dim) 

       

        # Define the decoder layers:
        self.fc2 = nn.Linear(latent_dim + label_embed_dims, conv_flat_dim) 
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv3 =nn.ConvTranspose2d(64, 32, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv4 =nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, stride=stride, padding=1, output_padding=1)
        

    def encode(self, x, label1, label2):

        # Encode image and concat it with labels
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        label1 = F.relu(self.em_label1(label1))
        label2 = F.relu(self.em_label2(label2))
        
        x = torch.cat((x, label1, label2), dim=1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def encode_guide(self, g):
        # Encode the guide 
        g = F.relu(self.guide_conv1(g))
        g = F.relu(self.guide_conv2(g))
        g = F.relu(self.guide_conv3(g))
        g = F.relu(self.guide_conv4(g))
        g = g.view(g.shape[0], -1)
        g = F.relu(self.fc_guide(g))
        
        return g


    def reparameterize(self, mean, logvar):
        # Reparameterize the latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z, label1, label2, guide):
        # Decode the latent space and concatenate with the label embeddings

        label1 = F.relu(self.em_label1(label1))
        label2 = F.relu(self.em_label2(label2))
        # label3 = F.relu(self.em_label3(label3))
        
        # cat = torch.cat((z, label1, label2, label3), dim=1)
        cat = torch.cat((z, label1, label2), dim=1)
        
        z = F.relu(self.fc2(cat))

        z = z.view(z.shape[0], self.conv_embed_shape[0], self.conv_embed_shape[1], self.conv_embed_shape[2])
        
        z = F.relu(self.deconv1(z))
        # print(z.shape)
        z = F.relu(self.deconv2(z))
        # print(z.shape)
        z = F.relu(self.deconv3(z))
        # print(z.shape)
        # z = torch.sigmoid(self.deconv4(z))
        z = torch.sigmoid(self.deconv4(z) + guide)
        # print(z.shape)
        return z

    def forward(self, x, label1, label2, guides):
        # Encode the input image and sample from the latent space
        mean, logvar = self.encode(x, label1, label2)
        z = self.reparameterize(mean, logvar)

        # Encode the guide
        # encoded_g = self.encode_guide(guide)

        # Add the z and the encoded guide
        embeddding = z

        # Decode the latent space to reconstruct the input image
        recon_x = self.decode(embeddding, label1, label2, guides)

        return recon_x, mean, logvar
    
    def sample(self, label1, label2, guides):
        batch_size = label1.shape[0]
        # sample a noise from normal distribution mean=0, std=I
        z = torch.randn((batch_size, self.latent_dim)).view(batch_size, -1).to(device=self.device)

        # Encode the guide
        # encoded_g = self.encode_guide(guide)

        # Add the z and the encoded guide
        embeddding = z
        
        return self.decode(embeddding, label1, label2, guides)

    def loss_fn(self, recon_x, x, mean, logvar, eps=0):
        # Compute the reconstruction loss
        # recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        square_diff = (recon_x - x) ** 2
        weight = torch.where(x > 0, 1.0, eps)
        square_loss = square_diff * weight
        recon_loss = torch.mean(square_loss)

        # Compute the KL divergence loss
        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = 0

        # Compute the total loss
        total_loss = recon_loss + self.beta*kl_loss

        return total_loss, recon_loss, kl_loss