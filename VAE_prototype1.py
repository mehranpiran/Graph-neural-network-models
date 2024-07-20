# GCN Variational Autoencoder

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc_mu = torch.nn.Linear(out_channels, out_channels)
        self.fc_logvar = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.fc_mu(x), self.fc_logvar(x)

class GCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = self.conv2(z, edge_index)
        return z

class VGAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAE, self).__init__()
        self.encoder = GCNEncoder(in_channels, out_channels)
        self.decoder = GCNDecoder(out_channels, in_channels)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, edge_index), mu, logvar

# Example usage with the Cora dataset from PyTorch Geometric
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(in_channels=dataset.num_features, out_channels=16).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    recon_x, mu, logvar = model(data.x, data.edge_index)
    loss = loss_function(recon_x, data.x, mu, logvar)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
recon_x, mu, logvar = model(data.x, data.edge_index)
print('Reconstruction loss:', F.mse_loss(recon_x, data.x, reduction='sum').item())
