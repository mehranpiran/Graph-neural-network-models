
## Variational Auto Encoder

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

class VAE(nn.Module):
    def __init__(self, input_dim, z_dim, split):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.split = split
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, z_dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def encode(self, x):
        return self.encoder(x)

    def encodeBatch(self, dataloader, device):
        indices = np.zeros((len(dataloader.dataset), self.split))
        output = np.zeros((len(dataloader.dataset), self.z_dim))

        model.eval()  # Set the model to evaluation mode
        scaler = GradScaler()

        with torch.no_grad():
            for i, (batch,) in enumerate(dataloader):
                batch = batch.to(device)
                with autocast():  # Mixed precision training
                    latent_embeddings = self.encode(batch).cpu().numpy()

                start_idx = i * dataloader.batch_size
                end_idx = start_idx + len(batch)
                output[start_idx:end_idx] = latent_embeddings

                # Quantize each split of the latent embeddings
                for j in range(self.split):
                    split_dim = self.z_dim // self.split
                    split_vectors = latent_embeddings[:, j * split_dim:(j + 1) * split_dim]
                    quantized_indices = [np.argmin(np.linalg.norm(codebook - vector, axis=1)) for vector in split_vectors]
                    indices[start_idx:end_idx, j] = quantized_indices

        return output, indices

# Example data and parameters
num_cells = 1000
num_features = 100
z_dim = 256  # Dimension of latent embedding D'
split = 4    # Number of splits M

# Randomly generate dataset
dataset = np.random.rand(num_cells, num_features)
tensor_dataset = TensorDataset(torch.tensor(dataset, dtype=torch.float32))
dataloader = DataLoader(tensor_dataset, batch_size=64, shuffle=False)

# Dummy codebook for demonstration
codebook = np.random.rand(1000, z_dim // split)

# Initialize model
vae_model = VAE(input_dim=num_features, z_dim=z_dim, split=split).to('cuda')

# Encode batch
latent_embeddings, feature_indices = vae_model.encodeBatch(dataloader, device='cuda')

print('Latent Embeddings:', latent_embeddings)
print('Feature Indices:', feature_indices)
