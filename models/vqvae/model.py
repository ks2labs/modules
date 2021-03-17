import math
import random
import numpy as np
from tqdm import trange
from collections import OrderedDict

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision.transforms import functional as TF


# always have a seed function to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# now coming to the challenging Vector Quantisation from the paper:
# https://papers.nips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf

class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, beta):
        super(VectorQuantizer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.codebook = nn.Embedding(vocab_size, hidden_dim)
        self.codebook.weight.data.uniform_(-1/vocab_size, 1/vocab_size)
        self.beta = beta
        
    def get_codebook_embeds(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.hidden_dim)
        
        # Calculate distances (a-b)^2 = a^2 + b^2 - 2ab
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) 
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
        )
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices
    
    def get_quantized(self, encoded_ids):
        codebook_embeds = torch.zeros(encoded_ids.shape[0], self.vocab_size, device=encoded_ids.device)
        codebook_embeds.scatter_(1, encoded_ids, 1)
        return torch.matmul(codebook_embeds, self.codebook.weight)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        encoded_ids = self.get_codebook_embeds(inputs)
        codebook_embeds = torch.zeros(encoded_ids.shape[0], self.vocab_size, device=encoded_ids.device)
        codebook_embeds.scatter_(1, encoded_ids, 1)
        # Quantize and unflatten
        quantized = torch.matmul(codebook_embeds, self.codebook.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(codebook_embeds, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, codebook_embeds

class Residual(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Residual, self).__init__()
        self._block = nn.Sequential(OrderedDict([
            ("relu_1", nn.ReLU(True)),
            ("conv_1", nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False)),
            ("relu_2", nn.ReLU(True)),
            ("conv_2", nn.Conv2d(hidden_dim, out_dim, 1, bias=False))
        ]))
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_dim, out_dim, n_residual_layers, hidden_dim):
        super().__init__()
        self.residual_blocks = nn.Sequential(OrderedDict([
            *[(f"residual_{i}", Residual(in_dim, out_dim, hidden_dim)) for i in range(n_residual_layers)],
            ("relu", nn.ReLU())
        ]))

    def forward(self, x):
        return self.residual_blocks(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_residual_layers, hidden_dim_residual):
        super().__init__()
        self._blocks = nn.Sequential(OrderedDict([
            ("conv_1", nn.Conv2d(in_dim, hidden_dim//2, 4, 2, padding=1)),
            ("relu_1", nn.ReLU()),
            ("conv_2", nn.Conv2d(hidden_dim//2, hidden_dim, 4, 2, padding=1)),
            ("relu_2", nn.ReLU()),
            ("conv_3", nn.Conv2d(hidden_dim, hidden_dim, 3, 1, padding=1)),
            ("residual", ResidualStack(hidden_dim, hidden_dim, n_residual_layers, hidden_dim_residual))
        ]))

    def forward(self, inputs):
        return self._blocks(inputs)
      
class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_residual_layers, hidden_dim_residual):
        super().__init__()
        self._blocks = nn.Sequential(OrderedDict([
            ("conv_1", nn.Conv2d(in_dim, hidden_dim, 3, 1, padding=1)),
            ("residual", ResidualStack(hidden_dim, hidden_dim, n_residual_layers, hidden_dim_residual)),
            ("conv_t_1", nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, padding=1)),
            ("relu_1", nn.ReLU()),
            ("conv_t_2", nn.ConvTranspose2d(hidden_dim//2, 3, 4, 2, padding=1))
        ]))

    def forward(self, inputs):
        return self._blocks(inputs)


class VQVAE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_residual_layers,
        hidden_dim_residual, 
        vocab_size,
        beta = 0.25
    ):
        super().__init__()
        
        self.encoder = Encoder(3, hidden_dim, n_residual_layers, hidden_dim_residual)
        self.pre_vq_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, 1)
        self.vq_vae = VectorQuantizer(vocab_size, hidden_dim, beta)
        self.decoder = Decoder(hidden_dim, hidden_dim,  n_residual_layers, hidden_dim_residual)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
    @property
    def _num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def image_to_encodings(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        encoding_indices = self.vq_vae.get_codebook_embeds(z)
        return encoding_indices.view(x.shape[0], -1)
    
    def encodings_to_image(self, x):
        B = x.shape[0]
        flat_x = x.view(-1, 1) # [B*N, 1]
        out = self.vq_vae.get_quantized(flat_x)
        
        # now we need to automatically determine the final output shape from the 
        total_params = out.shape[0] * out.shape[1]
        side_dim = int(math.sqrt(total_params / (B * self.hidden_dim)))
        out_shape = (B, side_dim, side_dim, self.hidden_dim)
        
        # reshape and change [B,H,W,C] -> [B,C,H,W] for conv layers
        out = out.view(out_shape).permute((0, 3, 1, 2))
        recon = self.decoder(out)
        return recon

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity


# define the dataset for training VAE
def transform(x):
    x = TF.resize(x, (32, 32))
    if np.random.random() > 0.5:
        x = TF.vflip(x)
    if np.random.random() > 0.5:
        x = TF.hflip(x)
    x = TF.to_tensor(x)
    x = TF.normalize(x, (0.5,0.5,0.5), (1.0,1.0,1.0))
    return x


class VAEDataset(Dataset):
    def __init__(self):
        cifar_train = datasets.CIFAR10("./", download = True, train = True)
        cifar_test = datasets.CIFAR10("./", download = True, train = False)
        cifar1_train = datasets.CIFAR100("./", download = True, train = True)
        cifar1_test = datasets.CIFAR100("./", download = True, train = False)
        
        self.data = [
          cifar_train, 
          cifar_test, 
          cifar1_train, 
          cifar1_test,
        ]
        
        lens = [len(x) for x in self.data]
        self.cumlen = np.cumsum(lens)
        
        stacked = np.vstack([
            cifar_train.data,
            cifar_test.data,
            cifar1_train.data,
            cifar1_test.data
        ])
        self.var = np.var(stacked / 255.)

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, i):
        ds = self.data[(self.cumlen > i).astype(int).argmax()] # first get the dataset
        x = ds[i % len(ds)][0] # then get the correct index in that dataset
        return transform(x)


# global flags
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
NUM_TRAINING_UPDATES = 30000
HIDDEN_DIM = 128
N_RESIDUAL_LAYERS = 2
HIDDEN_DIM_RESIDUAL = 64
VOCAB_SIZE = 512

def get_model():
    vqvae = VQVAE(
      hidden_dim=HIDDEN_DIM,
      n_residual_layers=N_RESIDUAL_LAYERS,
      hidden_dim_residual=HIDDEN_DIM_RESIDUAL,
      vocab_size=VOCAB_SIZE
    )
    return vqvae


def get_trained_model(path):
    model = get_model()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    # this will take a bit of time to calculate the data variance
    training_data = VAEDataset()
    set_seed(4)
    vqvae = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = vqvae.to(device)
    optimizer = optim.Adam(vqvae.parameters(), lr=LEARNING_RATE, amsgrad=False)
    training_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    vqvae.train()
    train_res_recon_error = []
    train_res_perplexity = []
    pbar = trange(NUM_TRAINING_UPDATES)
    for i in pbar:
        pbar.set_description(
          f"L:{np.mean(train_res_recon_error[-100:]):.4f} | P: {np.mean(train_res_perplexity[-100:]):.2f}"
        )
      
        data = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = vqvae(data)
        recon_error = F.mse_loss(data_recon, data) / training_data.var
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

    torch.save(vqvae.state_dict(), "./vqvae.pt")
