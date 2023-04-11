import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary

class CVAE(nn.Module):
    """CVAE"""

    def __init__(self, 
                input_shape, 
                num_label1, 
                num_label2, 
                num_label3,
                latent_dim=128,
                label_embed_dim=32,
                device='cpu'):
        super().__init__()

        self.show_summary = False # Edit

        self.latent_dim = latent_dim
        self.label_embed_dim = label_embed_dim
        self.device = device

        stride = 2
        padding = 1
        
        # Define the encoder layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=stride, padding=padding)

        self.conv_embed_shape = (256, 6, 5) # EDIT

        conv_flat_dim = self.conv_embed_shape[0] * self.conv_embed_shape[1] * self.conv_embed_shape[2]
        self.fc1 = nn.Linear(conv_flat_dim, 256) 

        # Define the label embedding layers
        self.em_label1 = nn.Embedding(num_label1, label_embed_dim)
        self.em_label2 = nn.Embedding(num_label2, label_embed_dim)
        self.em_label3 = nn.Embedding(num_label3, label_embed_dim)

        label_embed_dims = 3 * label_embed_dim

        # Define the mean and logvar layers
        self.fc_mean = nn.Linear(256 + label_embed_dims, latent_dim)
        self.fc_logvar = nn.Linear(256 + label_embed_dims, latent_dim)

        # Define the decoder layers:
        self.fc2 = nn.Linear(latent_dim + label_embed_dims, conv_flat_dim) 

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv3 =nn.ConvTranspose2d(64, 32, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv4 =nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, stride=stride, padding=1, output_padding=1)
        

    def encode(self, x, label1, label2, label3):
    #     if self.show_summary:
    #         print('======= Encode ======= ')
    #         print(x.shape)
    #         print(label1.shape)
    #         print(label2.shape)

    #         print('->>>> Convolution')

        # Encode image and concat it with labels
        x = F.relu(self.conv1(x))
        # if self.show_summary:
        #     print(x.shape)
        x = F.relu(self.conv2(x))
        # if self.show_summary:
        #     print(x.shape)
        x = F.relu(self.conv3(x))
        # if self.show_summary:
        #     print(x.shape)
        x = F.relu(self.conv4(x))
        # if self.show_summary:
        #     print(x.shape)

        
        flat = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(flat))

        # if self.show_summary:
        #     print('->>>> Flatten')
        #     print(flat.shape)
        #     print(x.shape)

        
        label1 = F.relu(self.em_label1(label1))
        label2 = F.relu(self.em_label2(label2))
        label3 = F.relu(self.em_label3(label3))

        # if self.show_summary:
        #     print('->>>> Embed')
        #     print(label1.shape)
        #     print(label2.shape)
        #     print(label3.shape)
        
        x = torch.cat((x, label1, label2, label3), dim=1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        # if self.show_summary:
        #     print('->>>> Concat')
        #     print(x.shape)
        #     print('->>>> Mean / Var')
        #     print(mean.shape)
        #     print(logvar.shape)
        
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # Reparameterize the latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z, label1, label2, label3):
        # Decode the latent space and concatenate with the label embeddings

        # if self.show_summary:
        #     print('======= Decode ======= ')
        #     print(z.shape)

        
        label1 = F.relu(self.em_label1(label1))
        label2 = F.relu(self.em_label2(label2))
        label3 = F.relu(self.em_label3(label3))
        
        cat = torch.cat((z, label1, label2, label3), dim=1)
        
        z = F.relu(self.fc2(cat))

        # if self.show_summary:
        #     print('->>>> Embed')
        #     print(label1.shape)
        #     print(label2.shape)
        #     print('->>>> Concat')
        #     print(cat.shape)
        #     print(z.shape)

        
        z = z.view(z.shape[0], self.conv_embed_shape[0], self.conv_embed_shape[1], self.conv_embed_shape[2])
        
        # if self.show_summary:
        #     print('->>>> To 4D')
        #     print(z.shape)
        #     print('->>>> Convolution Transpose')

        
        z = F.relu(self.deconv1(z))
        # if self.show_summary:
        #     print(z.shape)
        z = F.relu(self.deconv2(z))
        # if self.show_summary:
        #     print(z.shape)
        z = F.relu(self.deconv3(z))
        # if self.show_summary:
        #     print(z.shape)
        z = F.sigmoid(self.deconv4(z))
        # if self.show_summary:
        #     print(z.shape)

        return z

    def forward(self, x, label1, label2, label3):
        # Encode the input image and sample from the latent space
        mean, logvar = self.encode(x, label1, label2, label3)
        z = self.reparameterize(mean, logvar)
        
        # Decode the latent space to reconstruct the input image
        recon_x = self.decode(z, label1, label2, label3)



        # if self.show_summary:
        #     print('======= Foward ======= ')
        #     print(recon_x.shape)
        #     print(mean.shape)
        #     print(logvar.shape)

        return recon_x, mean, logvar
    
    def sample(self, label1, label2, label3):
        # sample a noise from normal distribution mean=0, std=I
        z = torch.randn(self.latent_dim).view(1, -1).to(device=self.device)
        return self.decode(z, label1, label2, label3)

    def loss_fn(self, recon_x, x, mean, logvar):
        # Compute the reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Compute the total loss
        total_loss = recon_loss + kl_loss

        return total_loss



if __name__ == "__main__":
    num_label1 = 2107
    num_label2 = 7
    num_label3 = 60
    input_shape = (1, 1, 96, 80)
    cvae = CVAE(input_shape, num_label1, num_label2, num_label3, device='cpu')
    # summary(cvae, 
    #         input_size=[input_shape, (1, ), (1, ), (1, )], 
    #         dtypes=[torch.float, torch.int, torch.int, torch.int], 
    #         device='cpu')