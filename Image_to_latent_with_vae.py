import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

# Custom dataset to load images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_transformed = self.transform(img)
        return img_transformed, img_path

# Read all image paths from the file
with open("/local/home/hhamidi/codes/diffusers/examples/text_to_image/img_path.txt") as f:
    all_images = f.readlines()

all_images = [x.strip() for x in all_images]

# Load the pretrained autoencoder models
model_path = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

# Define the transform
size = 512
transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Create the dataset and dataloader
dataset = ImageDataset(all_images, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=15)

# Prepare the model and dataloader for accelerator
vae, dataloader = accelerator.prepare(vae, dataloader)

# Process each batch of images
first_batch = True
for imgs, paths in tqdm(dataloader, desc="Processing batches"):
    # Convert grayscale images to 3-channel images
    imgs_3channel = imgs.repeat(1, 3, 1, 1)

    # Pass the images through the autoencoders
    with torch.no_grad():
        encoded_vae = vae.encode(imgs_3channel).latent_dist.sample()
        if first_batch:
            reconstructed_vae = vae.decode(encoded_vae).sample

        latent_numpy = encoded_vae.cpu().numpy()
        
        # Save the latent representations
        for i, latent in enumerate(latent_numpy):
            latent_path = paths[i].replace("physionet.org", "vae_latent") + "_latent.npy"
            latent_dir = os.path.dirname(latent_path)
            os.makedirs(latent_dir, exist_ok=True)
            np.save(latent_path, latent)

        # For the first batch, display the original and reconstructed images
        if first_batch:
            # Reverse normalization
            unnormalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
            reconstructed_unnormalized_vae = unnormalize(reconstructed_vae).clamp_(0, 1)

            fig, axes = plt.subplots(2, 8, figsize=(20, 5))
            for i in range(8):
                # Original image
                img = imgs_3channel[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 0.5) + 0.5  # Unnormalize
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title('Original')

                # Reconstructed image from VAE
                rec_img_vae = reconstructed_unnormalized_vae[i].cpu().numpy().transpose(1, 2, 0)
                axes[1, i].imshow(rec_img_vae, cmap='gray')
                axes[1, i].axis('off')
                axes[1, i].set_title('Reconstructed (VAE)')

            plt.show()
            first_batch = False

print("All images processed.")
