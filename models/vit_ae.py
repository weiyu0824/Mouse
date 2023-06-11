import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ViTEncoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 depth=16,
                 nhead=8):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x):
        # x.shape = (batch, depth, height, width)
        
        self.trans_enc(x)

        print(x.shape)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, input_shape, output_shape, dims=[128, 256, 512]):
        super().__init__()
        dims = [input_shape.shape[0]] + dims + [output_shape.shape[0]]
        mlp = []
        for i in range(len(dims)):
            mlp.append(nn.Linear(dims[i], dims[i+1]))
        self.dec = nn.ModuleList(mlp)

    def forward(self, x):
        return self.dec[x]

class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass

class VITAE(nn.Module):
    def __init__(self, input_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = ViTEncoder(d_model=(patch_size ** 2))
        # self.decoder = MLPDecoder()

    def forward(self, x, genes, times):
        batch_size = x.shape[0]
        x = x.view(-1, batch_size, self.patch_size ** 2)
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        # x = self.decoder(x)
        return x

