import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import cv2
import os
import numpy as np
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.siamese_network import SiameseNetwork

class SiameseDataset(torch.utils.data.Dataset):
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
        
        if img1 is None: img1 = np.zeros((224, 224), dtype=np.uint8)
        if img2 is None: img2 = np.zeros((224, 224), dtype=np.uint8)
            
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        img1 = np.expand_dims(img1, axis=0) # shape (1, 224, 224)
        img2 = np.expand_dims(img2, axis=0)
        
        label = row['label']
        return torch.from_numpy(img1), torch.from_numpy(img2), torch.tensor([label], dtype=torch.float32)

def analyze_bank(bank_name, results):
    print(f"Analyzing {bank_name}...", flush=True)
    bank_dir = os.path.join(PROJECT_ROOT, bank_name)
    csv_file = os.path.join(bank_dir, "pair_dataset.csv")
    model_path = os.path.join(bank_dir, "local_model.pth")
    
    if not os.path.exists(model_path) or not os.path.exists(csv_file):
        print(f"Files missing for {bank_name}")
        return

    dataset = SiameseDataset(csv_file, bank_dir)
    df = pd.read_csv(csv_file)
    gen_idx = df[df['label'] == 1].index.tolist()
    for_idx = df[df['label'] == 0].index.tolist()
    
    # Take 50 each
    indices = np.random.choice(gen_idx, min(50, len(gen_idx)), replace=False).tolist() + \
              np.random.choice(for_idx, min(50, len(for_idx)), replace=False).tolist()
    
    dataloader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False)
    
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    local_gen = []
    local_for = []
    
    with torch.no_grad():
        for i1, i2, l in dataloader:
            o1, o2 = model(i1, i2)
            dists = nn.functional.pairwise_distance(o1, o2).numpy()
            lbls = l.numpy().flatten()
            for d, lbl in zip(dists, lbls):
                if lbl == 1: local_gen.append(float(d))
                else: local_for.append(float(d))
                
    results[bank_name] = {
        "local": {
            "gen_mean": float(np.mean(local_gen)) if local_gen else 0,
            "for_mean": float(np.mean(local_for)) if local_for else 0,
            "gen_min": float(min(local_gen)) if local_gen else 0,
            "for_min": float(min(local_for)) if local_for else 0,
            "gen_max": float(max(local_gen)) if local_gen else 0,
            "for_max": float(max(local_for)) if local_for else 0
        }
    }
    
    # Global Model check
    global_model_path = os.path.join(PROJECT_ROOT, "server", "global_model.pth")
    if os.path.exists(global_model_path):
        model.load_state_dict(torch.load(global_model_path, map_location='cpu'))
        model.eval()
        global_gen = []
        global_for = []
        with torch.no_grad():
            for i1, i2, l in dataloader:
                o1, o2 = model(i1, i2)
                dists = nn.functional.pairwise_distance(o1, o2).numpy()
                lbls = l.numpy().flatten()
                for d, lbl in zip(dists, lbls):
                    if lbl == 1: global_gen.append(float(d))
                    else: global_for.append(float(d))
        results[bank_name]["global"] = {
            "gen_mean": float(np.mean(global_gen)) if global_gen else 0,
            "for_mean": float(np.mean(global_for)) if global_for else 0
        }

if __name__ == "__main__":
    results = {}
    analyze_bank("Bank1", results)
    analyze_bank("Bank2", results)
    analyze_bank("Bank3", results)
    
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Done. Results saved to analysis_results.json")
