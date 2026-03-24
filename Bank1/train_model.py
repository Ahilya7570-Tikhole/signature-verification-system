import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.siamese_network import SiameseNetwork

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        # label=1 means genuine pair, label=0 means forged/negative pair
        # contrastive loss:
        # L = label * D^2 + (1-label) * max(0, margin - D)^2
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SiameseDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img1_path = os.path.join(self.root_dir, row['image_path1'])
        img2_path = os.path.join(self.root_dir, row['image_path2'])
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None:
            img1 = np.zeros((224, 224), dtype=np.uint8)
        if img2 is None:
            img2 = np.zeros((224, 224), dtype=np.uint8)
            
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        img1 = np.expand_dims(img1, axis=0) # shape: (1, 224, 224)
        img2 = np.expand_dims(img2, axis=0)
        
        label = torch.tensor([row['label']], dtype=torch.float32)

        return torch.from_numpy(img1), torch.from_numpy(img2), label

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "pair_dataset.csv")
    
    dataset = SiameseDataset(csv_file=csv_file, root_dir=base_dir)
    # Using a subset for faster demonstration if the dataset is huge
    subset_indices = np.random.choice(len(dataset), min(2000, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    epochs = 30
    batch_size = 16
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Training on {len(subset_indices)} pairs for {epochs} epochs (Batch: {batch_size})...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (img1, img2, label) in enumerate(dataloader, 0):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}")
                running_loss = 0.0
                
    save_path = os.path.join(base_dir, "local_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
