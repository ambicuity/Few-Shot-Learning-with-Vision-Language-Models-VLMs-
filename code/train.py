import argparse
import torch
import clip
from tqdm import tqdm
import numpy as np
import os
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict()).double()
    return model

class DualAlignmentAdapter(torch.nn.Module):
    def __init__(self, in_features, hidden_dim=256, alpha=0.5):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, in_features)
        )
        self.alpha = alpha
    
    def forward(self, x, prototypes):
        # x: [batch, dim]
        # prototypes: [num_classes, dim]
        
        # Branch 1: Adapted features aligned to prototypes
        x_adapt = self.adapter(x) + x
        x_adapt = x_adapt / x_adapt.norm(dim=-1, keepdim=True)
        
        # Branch 2: Direct cosine sim with prototypes (Visual Branch)
        logits_proto = x @ prototypes.t()
        
        # Branch 3: Adapted similarity (Prompt/Semantic Branch approximation)
        logits_adapt = x_adapt @ prototypes.t()
        
        return self.alpha * logits_adapt + (1 - self.alpha) * logits_proto

def train(args):
    set_seed(args.seed)
    
    # Mocking data loading for GHA demonstration if real data not present
    # In a real run, this would load EuroSAT/MVTec
    print(f"Loading Model: {args.backbone}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device)
    dim = model.visual.output_dim
    
    n_shot = args.shots
    print(f"Device: {device}")
    
    # Real Data Loading: OxfordIIITPet
    print(f"Loading Dataset: OxfordIIITPet")
    from torchvision.datasets import OxfordIIITPet
    from torch.utils.data import DataLoader

    # Define transforms
    # CLIP's preprocess is usually a torchvision Transform
    
    root_dir = "./data"
    os.makedirs(root_dir, exist_ok=True)
    
    try:
        dataset = OxfordIIITPet(root=root_dir, split='trainval', transform=preprocess, download=True)
    except Exception as e:
        print(f"Failed to download dataset locally (might be expected in CI if network restricted): {e}")
        # Fallback to random for CI smoke test if network fails, BUT we want real data
        # In Modal, network is available.
        raise e

    print(f"Dataset Size: {len(dataset)}")
    
    # Extract Features
    # We'll extract all features first, then sample k-shot
    print("Extracting features (this may take a moment)...")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
            
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    # Few-Shot Sampling
    # Select N classes? OxfordPets has 37. We use all.
    classes = torch.unique(all_labels)
    num_classes = len(classes)
    n_shot = args.shots
    
    print(f"Sampling {n_shot}-shot support set for {num_classes} classes...")
    
    support_indices = []
    query_indices = []
    
    for c in classes:
        # Get indices for this class
        c_indices = (all_labels == c).nonzero(as_tuple=True)[0]
        c_indices = c_indices[torch.randperm(len(c_indices))]
        
        if len(c_indices) < n_shot + 1:
             # Skip classes with not enough data (unlikely for Pets)
             continue
             
        support_indices.append(c_indices[:n_shot])
        query_indices.append(c_indices[n_shot:])
        
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    support_features = all_features[support_indices].to(device).double()
    support_labels = all_labels[support_indices].to(device)
    
    # Subsample query to avoid OOM or too long eval
    # Use max 50 queries per class
    MAX_QUERY = 50
    q_subset = []
    for c in classes:
        c_q = query_indices[all_labels[query_indices] == c]
        if len(c_q) > MAX_QUERY:
            c_q = c_q[:MAX_QUERY]
        q_subset.append(c_q)
    query_indices = torch.cat(q_subset)
    
    query_features = all_features[query_indices].to(device).double()
    query_labels = all_labels[query_indices].to(device)
    
    # Compute Prototypes
    prototypes = []
    for c in range(num_classes):
        # We need to map class ID 0..36
        # support_labels are already correct indices
        p = support_features[support_labels == c].mean(0)
        p /= p.norm()
        prototypes.append(p)
    prototypes = torch.stack(prototypes).to(device)
    
    # Train DAPT
    adapter = DualAlignmentAdapter(dim).double().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, eps=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Training Adapter on Real Features...")
    for epoch in range(50):
        optimizer.zero_grad()
        logits = adapter(support_features, prototypes)
        loss = criterion(logits, support_labels)
        loss.backward()
        optimizer.step()
            
    # Eval
    with torch.no_grad():
        logits = adapter(query_features, prototypes)
        preds = logits.argmax(dim=1)
        acc = (preds == query_labels).float().mean()
        
    print(f"Seed {args.seed} | Shots {args.shots} | OxfordPets Accuracy: {acc.item():.4f}")
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    with open(f'results/res_seed{args.seed}_shot{args.shots}.txt', 'w') as f:
        f.write(f"{acc.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='ViT-B/32')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    train(args)
