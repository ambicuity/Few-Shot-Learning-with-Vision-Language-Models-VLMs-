import modal
import os
import sys

# Define the image with necessary dependencies
# We use a standard Debian slim image and install our requirements
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
        "git+https://github.com/openai/CLIP.git"
    )
    .add_local_dir("code", remote_path="/root/code")
)

app = modal.App("few-shot-vlm-research")

# Mount is handled via image definition
# code_mount = modal.Mount.from_local_dir("code", remote_path="/root/code")

@app.function(
    image=image,
    gpu="any",  # Request any available GPU
    timeout=600, # 10 minutes max
)
def run_experiment(seed: int, shots: int):
    import sys
    import os
    
    # Add code directory to path so imports work if needed
    sys.path.append("/root/code")
    
    # Import train directly since we are ensuring compatibility
    try:
        import train
    except ImportError:
        # Fallback if path handling differs
        from code import train
    
    class Args:
        def __init__(self, s, sh):
            self.seed = s
            self.shots = sh
            self.backbone = 'ViT-B/32' # Default
            
    args = Args(seed, shots)
    
    print(f"Running Experiment on Modal: Seed={seed}, Shots={shots}")
    
    # Run training
    # Note: train.py writes to 'results/'. In Modal, this is local to the container.
    # We need to return the result content or handle artifacts.
    # For now, we will modify train.py to return value or just print.
    # But since we can't easily modify the *imported* train.py deeply without reloading,
    # let's assume train.py prints to stdout and writes a file.
    
    # To persist results, we could use a Volume, or just return the accuracy float.
    # Let's import train and run it.
    
    # We need to make sure train.py doesn't sys.exit().
    # It does not.
    
    # However, train.py writes to 'results/...'.
    # We'll read that file back and return the content.
    try:
        # Check if train is a module or function
        if hasattr(train, 'train'):
            train.train(args)
        else:
            # If imported as function
            train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
        
    # Read result
    res_file = f'results/res_seed{args.seed}_shot{args.shots}.txt'
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            return f.read().strip()
    return "0.0"

@app.local_entrypoint()
def main():
    # Run 3 seeds in parallel
    seeds = [1, 2, 3]
    shots = 16
    
    results = list(run_experiment.map(seeds, kwargs={"shots": shots}))
    
    # Save results locally for the GHA to pick up
    os.makedirs("results", exist_ok=True)
    for seed, acc in zip(seeds, results):
        filename = f"results/res_seed{seed}_shot{shots}.txt"
        with open(filename, "w") as f:
            f.write(acc)
        print(f"Seed {seed}: {acc}")

