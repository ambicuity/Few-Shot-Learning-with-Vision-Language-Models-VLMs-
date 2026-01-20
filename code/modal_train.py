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

# Use a volume to persist the dataset across seeds so we don't redownload 3 times
data_volume = modal.Volume.from_name("oxford-pets-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="any", 
    timeout=1800, # Increased to 30 mins for download + extraction
    volumes={"/root/data": data_volume} # Mount volume at /root/data
)
def run_experiment(seed: int, shots: int):
    import sys
    import os
    
    # Add code directory to path
    sys.path.append("/root/code")
    
    # Ensure data directory exists (handled by mount, but just in case)
    os.makedirs("/root/data", exist_ok=True)
    
    # Import train directly
    try:
        import train
    except ImportError:
        from code import train
    
    class Args:
        def __init__(self, s, sh):
            self.seed = s
            self.shots = sh
            self.backbone = 'ViT-B/32'
            # Patch dataset root in train.py? 
            # We will rely on train.py using "./data" or we modify it to "data".
            # train.py uses "./data". 
            # We are in /root. so "./data" -> /root/data. which is the volume. Perfect.
            
    args = Args(seed, shots)
    
    print(f"Running Experiment on Modal: Seed={seed}, Shots={shots}")
    
    try:
        if hasattr(train, 'train'):
            train.train(args)
        else:
            train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
        
    res_file = f'results/res_seed{args.seed}_shot{args.shots}.txt'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            return f.read().strip()
    return "0.0"

@app.local_entrypoint()
def main():
    seeds = [1, 2, 3]
    shots = 16
    results = list(run_experiment.map(seeds, kwargs={"shots": shots}))
    
    os.makedirs("results", exist_ok=True)
    for seed, acc in zip(seeds, results):
        filename = f"results/res_seed{seed}_shot{shots}.txt"
        with open(filename, "w") as f:
            f.write(acc)
        print(f"Seed {seed}: {acc}")
