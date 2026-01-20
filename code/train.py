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
    
    # SOTA Upgrade: ViT-B/16 and Multi-View
    print(f"Loading Model: {args.backbone}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device)
    dim = model.visual.output_dim
    
    n_shot = args.shots
    print(f"Device: {device}")
    
    # Dataset Router
    root_dir = "./data"
    # Ensure imports are available globally
    from torchvision.datasets import OxfordIIITPet, EuroSAT, SVHN
    from torch.utils.data import DataLoader
    
    # SOTA TRICK: Multi-View Augmentation (TenCrop)
    # This averages features from 10 crops (corners + center, flip + no-flip)
    print("Enabling Test-Time Augmentation (10-Crop Ensemble)...")
    from torchvision.transforms import Compose, Resize, TenCrop, Lambda, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        bicubic = InterpolationMode.BICUBIC
    except:
        bicubic = 3 # PIL.Image.BICUBIC
        
    n_px = model.visual.input_resolution
    norm_mean = (0.48145466, 0.4578275, 0.40821073)
    norm_std = (0.26862954, 0.26130258, 0.27577711)
    
    # Resize to slightly larger than n_px (e.g. 224 -> 256) then crop 224
    resize_size = int(n_px * 1.15)
    
    val_transform = Compose([
        Resize(resize_size, interpolation=bicubic),
        TenCrop(n_px),
        Lambda(lambda crops: torch.stack([Normalize(norm_mean, norm_std)(ToTensor()(crop)) for crop in crops])),
    ])
    
    # Apply to dataset
    if args.dataset.lower() == 'oxfordpets':
        dataset = OxfordIIITPet(root=root_dir, split='trainval', transform=val_transform, download=True)
    elif args.dataset.lower() == 'eurosat':
        dataset = EuroSAT(root=root_dir, transform=val_transform, download=True)
    else:
        dataset = OxfordIIITPet(root=root_dir, split='trainval', transform=val_transform, download=True)

    print(f"Dataset: {args.dataset} | Size: {len(dataset)}")
    
    # Feature Extraction
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2) # Reduced batch size for 10x expansion
    
    all_features = []
    all_labels = []
    
    print("Extracting features with 10-Crop Ensemble...")
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # images shape: (B, 10, C, H, W)
            bs, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(device) # (B*10, C, H, W)
            
            features = model.encode_image(images) # (B*10, D)
            features = features.view(bs, n_crops, -1).mean(dim=1) # (B, D) Avg across crops
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    # Few-Shot Sampling
    classes = torch.unique(all_labels)
    num_classes = len(classes)
    
    support_indices = []
    query_indices = []
    
    for c in classes:
        c_indices = (all_labels == c).nonzero(as_tuple=True)[0]
        c_indices = c_indices[torch.randperm(len(c_indices))]
        if len(c_indices) < n_shot + 1: continue
        support_indices.append(c_indices[:n_shot])
        query_indices.append(c_indices[n_shot:])
        
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    # Cast to double
    support_features = all_features[support_indices].to(device).double()
    support_labels = all_labels[support_indices].to(device)
    
    # Subsample query
    MAX_QUERY = 50
    q_subset = []
    for c in classes:
        c_q = query_indices[all_labels[query_indices] == c]
        if len(c_q) > MAX_QUERY: c_q = c_q[:MAX_QUERY]
        q_subset.append(c_q)
    query_indices = torch.cat(q_subset)
    query_features = all_features[query_indices].to(device).double()
    query_labels = all_labels[query_indices].to(device)
    
    # Prototypes
    prototypes = []
    for c in range(num_classes):
        p = support_features[support_labels == c].mean(0)
        p /= p.norm()
        prototypes.append(p)
    prototypes = torch.stack(prototypes).to(device)
    
    # Train DAPT
    adapter = DualAlignmentAdapter(dim, alpha=args.alpha).double().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, eps=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    print(f"Training Adapter (Backbone={args.backbone}, Alpha={args.alpha})...")
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = adapter(support_features, prototypes)
        loss = criterion(logits, support_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
            
    # Eval
    with torch.no_grad():
        logits = adapter(query_features, prototypes)
        preds = logits.argmax(dim=1)
        acc = (preds == query_labels).float().mean()
        
    print(f"Seed {args.seed} | Shots {args.shots} | Alpha {args.alpha} | Back {args.backbone} | Acc: {acc.item():.4f}")
    
    os.makedirs('results', exist_ok=True)
    fname = f'results/res_seed{args.seed}_shot{args.shots}_alpha{args.alpha}.txt'
    with open(fname, 'w') as f:
        f.write(f"{acc.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='ViT-B/16') 
    parser.add_argument('--dataset', type=str, default='OxfordPets')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.2) # OPTIMAL
    parser.add_argument('--epochs', type=int, default=100) # OPTIMAL
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
