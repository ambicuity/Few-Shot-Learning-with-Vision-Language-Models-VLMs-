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

@app.function(
    image=image,
    gpu="any", 
    timeout=1800,
    volumes={"/root/data": data_volume}
)
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
    # Grid Search Strategy
    seeds = [1, 2] # Reduced to 2 seeds for speed in demo
    alphas = [0.2, 0.5, 0.8] # Sweep alpha
    epochs = 100 # Improved training duration
    shots = 16
    
    # Create grid
    configs = []
    for s in seeds:
        for a in alphas:
            configs.append((s, shots, a, epochs))
            
    print(f"Launching {len(configs)} parallel experiments for grid search...")
    
    # Starmap over configs
    # We unpack the configs tuple
    results = list(run_experiment.starmap(configs))
    
    os.makedirs("results", exist_ok=True)
    
    # Process results
    best_acc = 0.0
    best_cfg = None
    
    for (seed, shot, alpha, epoch), acc in zip(configs, results):
        filename = f"results/res_seed{seed}_shot{shot}_alpha{alpha}.txt"
        with open(filename, "w") as f:
            f.write(acc)
        print(f"Config [Seed={seed}, Alpha={alpha}]: {acc}")
        
        try:
            if float(acc) > best_acc:
                best_acc = float(acc)
                best_cfg = (alpha, epoch)
        except:
            pass
            
    print(f"Search Complete. Best Accuracy: {best_acc}")
    print(f"Recommended Config: Alpha={best_cfg[0]}, Epochs={best_cfg[1]}")
