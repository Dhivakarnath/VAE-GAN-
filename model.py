import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAEGAN(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, z_dim):
        super(ConditionalVAEGAN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_encode = nn.Linear(hidden_dim * 4 * (input_dim[1] // 8) * (input_dim[2] // 8) + cond_dim, z_dim * 2)

        self.fc_decode = nn.Linear(z_dim + cond_dim, hidden_dim * 4 * (input_dim[1] // 8) * (input_dim[2] // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, cond):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cond], dim=1)
        x = self.fc_encode(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        z = self.fc_decode(z)
        z = z.view(z.size(0), -1, 4, 4)  # Adjust dimensions based on the output shape of `fc_decode`
        return self.decoder(z)

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, cond)
        return x_recon, mu, logvar
