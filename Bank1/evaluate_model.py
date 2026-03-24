import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from train_model import SiameseNetwork, SiameseDataset

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "pair_dataset.csv")
    model_path = os.path.join(base_dir, "local_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    # Load dataset
    dataset = SiameseDataset(csv_file=csv_file, root_dir=base_dir)
    # Use a subset for evaluation (e.g., 500 pairs)
    subset_indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=False)
    
    # Load model
    model = SiameseNetwork().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    correct = 0
    total = 0
    distances = []
    labels = []

    print(f"Evaluating on {len(subset)} pairs...")
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            
            # Calculate Euclidean distance
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            
            distances.extend(euclidean_distance.cpu().numpy())
            labels.extend(label.cpu().numpy().flatten())

    # We need a threshold to decide if the pair is same or different
    # In contrastive loss, label=1 means "same" (genuine), label=0 means "different" (forged)
    # Typically, small distance = same, large distance = different
    
    # Try multiple thresholds to find a reasonable one, or just pick 1.0 as a baseline
    best_threshold = 1.0
    
    predictions = [1 if d < best_threshold else 0 for d in distances]
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = (correct / len(labels)) * 100

    print(f"\nResults with threshold {best_threshold}:")
    print(f"Total pairs: {len(labels)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # More metrics
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

if __name__ == '__main__':
    evaluate()
