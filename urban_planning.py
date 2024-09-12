import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import ast

# Model Definition
class ConditionalVAEGAN(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, z_dim):
        super(ConditionalVAEGAN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim, kernel_size=4, stride=2, padding=1), # 64x64xhidden_dim
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1), # 32x32x(hidden_dim*2)
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1), # 16x16x(hidden_dim*4)
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1), # 8x8x(hidden_dim*8)
            nn.ReLU()
        )
        self.flatten_dim = hidden_dim * 8 * (input_dim[1] // 16) * (input_dim[2] // 16)
        self.fc_encode = nn.Linear(self.flatten_dim + cond_dim, z_dim * 2)

        self.fc_decode = nn.Linear(z_dim + cond_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1), # 16x16x(hidden_dim*4)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1), # 32x32x(hidden_dim*2)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1), # 64x64x(hidden_dim)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim[0], kernel_size=4, stride=2, padding=1), # 128x128xinput_dim
            nn.Sigmoid()
        )

    def encode(self, x, cond):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = torch.cat([x, cond], dim=1) # Concatenate with condition vector
        x = self.fc_encode(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z = torch.cat([z, cond], dim=1) # Concatenate with condition vector
        z = self.fc_decode(z)
        z = z.view(z.size(0), -1, 8, 8) # Reshape to match the output of decoder layers
        return self.decoder(z)

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, cond)
        return x_recon, mu, logvar

# Custom Dataset Definition
class CustomUrbanDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_file)
        self.metadata['SocioEconomic_constraints'] = self.metadata['SocioEconomic_constraints'].map({
            'low': 0, 'medium': 1, 'high': 2
        }).astype(int)
        self.transform = transform

        # Determine the expected size of zoning vectors
        self.zoning_size = None
        self._determine_zoning_size()

        # Ensure all conditions are of the same size
        self.expected_cond_size = self.zoning_size + 4  # Add other conditions size

    def _determine_zoning_size(self):
        # Determine zoning vector size from the first entry in the metadata
        if not self.metadata.empty:
            example_zoning_data = ast.literal_eval(self.metadata.iloc[0]['Zoning_data'])
            self.zoning_size = len(example_zoning_data)

    def _pad_or_truncate(self, tensor, size):
        if tensor.size(0) < size:
            padding = torch.zeros(size - tensor.size(0))
            tensor = torch.cat([tensor, padding], dim=0)
        elif tensor.size(0) > size:
            tensor = tensor[:size]
        return tensor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx]['image_name']
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        zoning_data = ast.literal_eval(self.metadata.iloc[idx]['Zoning_data'])
        zoning_vector = torch.tensor(list(zoning_data.values()), dtype=torch.float32)

        # Pad or truncate zoning_vector to ensure it matches the expected size
        if self.zoning_size is not None:
            zoning_vector = self._pad_or_truncate(zoning_vector, self.zoning_size)

        building_density = torch.tensor([self.metadata.iloc[idx]['Building_density']], dtype=torch.float32)
        traffic_flow = torch.tensor([self.metadata.iloc[idx]['Traffic_flow']], dtype=torch.float32)
        environmental_constraints = torch.tensor([self.metadata.iloc[idx]['Environmental_constraints']], dtype=torch.float32)
        socioeconomic_constraints = torch.tensor([self.metadata.iloc[idx]['SocioEconomic_constraints']], dtype=torch.float32)

        conditions = torch.cat([zoning_vector, building_density, traffic_flow, environmental_constraints, socioeconomic_constraints], dim=0)

        # Ensure condition tensor size matches expected size
        if conditions.size(0) != self.expected_cond_size:
            raise ValueError(f"Condition vector size mismatch: expected {self.expected_cond_size}, got {conditions.size(0)}")

        return image, conditions

# Utility functions for VAE-GAN
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = torch.nn.MSELoss()(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

def gan_loss(real_pred, fake_pred):
    real_labels = torch.ones_like(real_pred)
    fake_labels = torch.zeros_like(fake_pred)
    criterion = torch.nn.BCEWithLogitsLoss()
    real_loss = criterion(real_pred, real_labels)
    fake_loss = criterion(fake_pred, fake_labels)
    return real_loss + fake_loss

def generate_image(model, condition, device='cpu'):
    model.eval()
    with torch.no_grad():
        condition = condition.unsqueeze(0).to(device)
        mu, logvar = model.encode(torch.zeros((1, 3, 128, 128)).to(device), condition)
        z = model.reparameterize(mu, logvar)
        generated_image = model.decode(z, condition)
        generated_image = generated_image.squeeze().cpu()
        generated_image = torch.clamp(generated_image, 0, 1)
        generated_image = transforms.ToPILImage()(generated_image)
    return generated_image

# Streamlit App
st.title("Urban Planning with Conditional VAE-GAN")

# Upload buttons for images and metadata
st.header("Upload Images and Metadata")
image_dir = st.file_uploader("Upload Images (jpg, png, jpeg)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])

# Ensure the user has uploaded both images and metadata before proceeding
if image_dir and metadata_file:
    st.success("Images and Metadata uploaded successfully.")

    # Save uploaded images to a folder
    img_folder_path = "uploaded_images"
    os.makedirs(img_folder_path, exist_ok=True)
    
    for img_file in image_dir:
        with open(os.path.join(img_folder_path, img_file.name), "wb") as f:
            f.write(img_file.getbuffer())

    # Save metadata file
    metadata_path = "metadata.csv"
    with open(metadata_path, "wb") as f:
        f.write(metadata_file.getbuffer())
    
    # Display training button
    if st.button("Train Model"):
        st.write("Training in progress...")

        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        # Load dataset
        dataset = CustomUrbanDataset(image_dir=img_folder_path, metadata_file=metadata_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConditionalVAEGAN(input_dim=(3, 128, 128), cond_dim=dataset.expected_cond_size, hidden_dim=256, z_dim=128).to(device)

        # Training parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 150
        beta = 1.0

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, conditions in dataloader:
                images = images.to(device)
                conditions = conditions.to(device)

                # Forward pass and loss calculation
                x_recon, z_mu, z_logvar = model(images, conditions)
                vae_loss_val = vae_loss(x_recon, images, z_mu, z_logvar, beta)
                # GAN loss is not used here, as the model does not currently output it
                loss = vae_loss_val

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            st.write(f'Epoch [{epoch+1}/{num_epochs}]')

        st.success("Training complete!")

        # Display generate image button
        st.header("Generate Image")
        cond_input = st.text_input("Enter Condition Vector (comma-separated)")

        if st.button("Generate"):
            try:
                cond_vector = list(map(float, cond_input.split(',')))
                if len(cond_vector) != dataset.expected_cond_size:
                    st.error(f"Condition vector size mismatch: expected {dataset.expected_cond_size}, got {len(cond_vector)}")
                else:
                    cond_tensor = torch.tensor(cond_vector, dtype=torch.float32)
                    generated_img = generate_image(model, cond_tensor, device)
                    st.image(generated_img, caption="Generated Image")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("Please upload images and metadata to proceed.")
