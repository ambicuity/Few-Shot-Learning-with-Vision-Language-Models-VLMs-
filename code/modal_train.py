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
    # config: (seed, shots, alpha, epochs, visualize, dataset)
    print(f"DEBUG: Processing config: {config}")
    seed, shots, alpha, epochs, visualize, dataset_name = config
    
    class Args:
        def __init__(self, s, sh, a, e, vis=False, ds='OxfordPets'):
            self.seed = s
            self.shots = sh
            self.alpha = a
            self.epochs = e
            self.lr = 1e-3
            # UPGRADE: SOTA Backbone
            self.backbone = 'ViT-B/16'
            self.dataset = ds
            self.visualize = vis
            
    args = Args(seed, shots, alpha, epochs, visualize, dataset_name)
    
    print(f"Running on Modal: Dataset={args.dataset}, Seed={seed}")
    
    try:
        if hasattr(train, 'train'):
            train.train(args)
        else:
            train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        return "0.0"
        
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
    datasets = ['OxfordPets', 'EuroSAT'] # Run both supported datasets
    
    # Pass TUPLES to map: (seed, shots, alpha, epochs, visualize, dataset)
    configs = []
    for ds in datasets:
        for s in seeds:
            vis = (s == 1) # Visualize only seed 1 per dataset
            configs.append((s, shots, alpha, epochs, vis, ds))
    
    print(f"Launching 5-Seed Robustness Verification for {datasets}...")
    
    # Run in parallel
    results = list(run_experiment.map(configs))
    
    # Aggregate results
    final_report = {}
    
    # Unpack results back to structure
    idx = 0
    for ds in datasets:
        ds_accs = []
        print(f"\n--- Results for {ds} ---")
        for s in seeds:
            res = results[idx]
            idx += 1
            try:
                acc = float(res)
            except:
                acc = 0.0
            ds_accs.append(acc)
            print(f"Seed {s}: {acc:.4f}")
        
        mean_acc = np.mean(ds_accs)
        std_acc = np.std(ds_accs)
        final_report[ds] = (mean_acc, std_acc)
        print(f"LaTeX: {mean_acc*100:.1f} \\pm {std_acc*100:.1f}")

    print(f"\n=== FINAL SUMMARY ===")
    for ds, (m, s) in final_report.items():
        print(f"{ds}: {m*100:.1f} +/- {s*100:.1f}")
