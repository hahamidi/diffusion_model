# CUDA_VISIBLE_DEVICES=0,1,3 python text_to_embedding.py
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

def generate_and_save_embeddings(dataloader):
    """
    Process a dataloader containing batches of sentence and path pairs to generate embeddings and save them to specified paths.

    Args:
    dataloader (DataLoader): A DataLoader object that yields batches of tuples (sentences, save_paths).
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for batch in tqdm(dataloader, desc="Processing batches"):
            sentences, save_paths = batch
            # Encode the sentences to get token ids and attention masks
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

            # Pass the inputs to the model and get the last hidden state
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

            # Average the embeddings over the sequence length to get a single vector for each sentence
            sentence_embeddings = embeddings.mean(dim=1)
            # Process each sentence embedding in the batch
            for i, sentence_embedding in enumerate(sentence_embeddings):
                # Move embedding back to CPU for saving
                sentence_embedding = sentence_embedding.cpu()

                # Ensure the directory exists
                dir_path = os.path.dirname(save_paths[i])
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # Save the embeddings as a tensor to the specified path
                torch.save(sentence_embedding, f'{save_paths[i]}.pt')

                # To save as a numpy array, convert it first
                sentence_embedding_np = sentence_embedding.numpy()
                np.save(f'{save_paths[i]}.npy', sentence_embedding_np)

# Read data for DataLoader
with open("/local/home/hhamidi/t_dif/diffusers/examples/text_to_image/dataset/sentence.txt", "r") as f:
    sentences = f.read().splitlines()

with open("/local/home/hhamidi/t_dif/diffusers/examples/text_to_image/dataset/img_path.txt", "r") as f:
    img_paths = f.read().splitlines()

base_path= "/local/home/hhamidi/t_dif/diffusers/examples/text_to_image/dataset/"
img_paths = [ base_path + item for item in img_paths]
# Combine sentences and image paths into a list of tuples
data = list(zip(sentences, img_paths))

# Create DataLoader
dataloader = DataLoader(data, batch_size=16 * torch.cuda.device_count(), shuffle=False)  # Adjust batch_size as needed based on the number of GPUs

# Call the function
generate_and_save_embeddings(dataloader)
