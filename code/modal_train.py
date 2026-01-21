import modal
import os
import sys

# Define the image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "ftfy",
        "regex",
        "tqdm",
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "git+https://github.com/openai/CLIP.git"
    )
    .add_local_dir("code", remote_path="/root/code")
)

app = modal.App("few-shot-vlm-research")

# Use a volume to persist the dataset
data_volume = modal.Volume.from_name("oxford-pets-data", create_if_missing=True)


@app.function(image=image, gpu="any", timeout=1800, volumes={"/root/data": data_volume})
def run_experiment(config):
    # Setup imports inside the container
    import sys
    import os
    sys.path.append("/root/code")
    os.makedirs("/root/data", exist_ok=True)
    
    try:
        import train
    except ImportError:
        from code import train

    # Fix: Unpack tuple configuration
    print(f"DEBUG: Processing config: {config}")
    seed, shots, alpha, epochs, visualize = config
    
    class Args:
        def __init__(self, s, sh, a, e, vis=False):
            self.seed = s
            self.shots = sh
            self.alpha = a
            self.epochs = e
            self.lr = 1e-3
            # UPGRADE: SOTA Backbone
            self.backbone = 'ViT-B/16'
            self.dataset = 'OxfordPets'
            self.visualize = vis
            
    args = Args(seed, shots, alpha, epochs, visualize)
    
    print(f"Running SOTA-Optimization on Modal: Back={args.backbone}, Seed={seed}")
    
    try:
        if hasattr(train, 'train'):
            train.train(args)
        else:
            train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
        
    res_file = f'results/res_seed{args.seed}_shot{args.shots}_alpha{args.alpha}.txt'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            return f.read().strip()
    return "0.0"

@app.local_entrypoint()
def main():
    import numpy as np
    # Fix 1: Run 5 Seeds for robust Std Dev
    seeds = [1, 2, 3, 4, 5]
    alpha = 0.2
    epochs = 100
    shots = 16
    
    # Pass TUPLES to map: (seed, shots, alpha, epochs, visualize)
    configs = []
    for s in seeds:
        vis = (s == 1) # Visualize only seed 1
        configs.append((s, shots, alpha, epochs, vis))
    
    print(f"Launching 5-Seed Robustness Verification (Alpha={alpha})...")
    
    accuracies = []
    
    # Run in parallel using .map() because run_experiment takes a single tuple argument
    results = list(run_experiment.map(configs))
    
    for i, res in enumerate(results):
        try:
            acc = float(res)
        except:
            acc = 0.0
        accuracies.append(acc)
        print(f"Seed {seeds[i]}: {acc:.4f}")
        
    # Stats
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n=== FINAL ROBUSTNESS REPORT ===")
    print(f"Config: ViT-B/16, Alpha={alpha}, 16-Shot + TTA")
    print(f"Accuracies: {accuracies}")
    # Format for LaTeX: Mean +/- Std
    print(f"LaTeX Format: {mean_acc*100:.1f} \\pm {std_acc*100:.1f}")
    
    with open("robustness_results.txt", "w") as f:
        f.write(f"MEAN:{mean_acc:.4f}\nSTD:{std_acc:.4f}\nRAW:{','.join(map(str, accuracies))}")
