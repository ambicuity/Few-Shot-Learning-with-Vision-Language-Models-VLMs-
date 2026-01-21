import os
import subprocess
import zipfile
import sys

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(base_dir, 'paper')
    
    if not os.path.exists(paper_dir):
        print(f"Error: Paper directory not found at {paper_dir}")
        return

    os.chdir(paper_dir)
    print(f"Working directory: {os.getcwd()}")

    # Files to include
    files_to_include = [
        'main.tex',
        'literature_review.tex',
        'method.tex',
        'proofs.tex',
        'datasets.tex',
        'baselines.tex',
        'results.tex',
        'analysis.tex',
        'ethics.tex',
        'conclusion.tex',
        'main.bbl', # Critical
        'references.bib' # Optional but good practice
    ]

    # 1. Compile (Try to generate .bbl)
    print("Attempting to compile paper to generate .bbl file...")
    if run_command("pdflatex -interaction=nonstopmode main.tex") and \
       run_command("bibtex main") and \
       run_command("pdflatex -interaction=nonstopmode main.tex"):
        print("Compilation successful.")
    else:
        print("WARNING: Compilation failed or pdflatex/bibtex not found.")
        print("Please ensure you have a LaTeX distribution installed if you need to generate the .bbl file.")

    # 2. Check for .bbl
    if not os.path.exists('main.bbl'):
        print("CRITICAL ERROR: 'main.bbl' file is missing. arXiv requires this file for BibTeX submissions.")
        print("Cannot create a valid submission archive without it.")
        # Proceeding anyway usually results in rejection, but let's zip what we have with a warning.
    
    # 3. Create Zip
    zip_filename = 'submission.zip'
    print(f"Creating {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in files_to_include:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"Added {file}")
                else:
                    if file == 'main.bbl':
                        print(f"SKIPPING MISSING CRITICAL FILE: {file}")
                    else:
                        print(f"Warning: File {file} not found.")
        
        print(f"\nArchive created at: {os.path.abspath(zip_filename)}")
        if not os.path.exists('main.bbl'):
             print("\nIMPORTANT: You MUST generate 'main.bbl' and add it to this archive before uploading to arXiv.")
    except Exception as e:
        print(f"Error creating zip file: {e}")

if __name__ == "__main__":
    main()
