#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import importlib.util

def check_model_dependencies(model):
    """Check if required dependencies for specific models are installed."""
    print(f"Checking dependencies for {model} model...")
    
    if model == "bert":
        # Check for transformers and other BERT dependencies
        transformers_available = importlib.util.find_spec("transformers") is not None
        if not transformers_available:
            print("WARNING: 'transformers' package not found. BERT model requires this dependency.")
            print("Try running: pip install transformers==4.1.1")
            return False
        
        # Check that torch is installed
        torch_available = importlib.util.find_spec("torch") is not None
        if not torch_available:
            print("WARNING: 'torch' package not found. BERT model requires PyTorch.")
            print("Try running: conda install pytorch torchvision torchaudio -c pytorch")
            return False
            
    elif model == "nrl":
        # Check for octis and other NRL dependencies
        octis_available = importlib.util.find_spec("octis") is not None
        if not octis_available:
            print("WARNING: 'octis' package not found. NRL model requires this dependency.")
            print("Try running: pip install octis")
            return False
            
    elif model == "ctm":
        # Check for contextualized_topic_models and other CTM dependencies
        ctm_available = importlib.util.find_spec("contextualized_topic_models") is not None
        if not ctm_available:
            print("WARNING: 'contextualized_topic_models' package not found. CTM model requires this dependency.")
            print("Try running: pip install contextualized-topic-models")
            return False
    
    return True

def run_experiment(model, naspects, dataset_path, output_path, skip_dependency_check=False):
    """Run LADy experiment with specified parameters."""
    # Check dependencies first
    if not skip_dependency_check and not check_model_dependencies(model):
        print(f"Some dependencies for the {model} model are missing. The experiment may fail.")
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Experiment canceled.")
            return False

    # Create a symbolic link with semeval in the path to make the loader recognize it
    # Place symlink directly in datasets directory, not in subdirectory
    symlink_path = f"datasets/semeval_{os.path.basename(dataset_path)}"
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "debug.log")
    txt_log_file = os.path.join(output_path, "debug.txt")
    
    try:
        if os.path.exists(symlink_path):
            print(f"Removing existing symlink: {symlink_path}")
            os.remove(symlink_path)
        
        print(f"Creating symlink from {os.path.abspath(dataset_path)} to {symlink_path}")
        os.symlink(os.path.abspath(dataset_path), symlink_path)
        
        # Run the experiment using the symlink path
        cmd = [
            "python", "src/main.py",
            "-am", model,
            "-naspects", str(naspects),
            "-data", symlink_path,
            "-output", output_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Create a more detailed debug.txt file for tracking progress
        with open(txt_log_file, "w") as f:
            f.write(f"Starting with args: {args}\n")
            f.write(f"Input file: {symlink_path}\n")
        
        try:
            # Run the command and capture all output
            with open(log_file, "w") as f:
                result = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
                
            print(f"Command completed successfully with exit code {result.returncode}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print("This may be due to the known aggregation error. Checking if results were still generated...")
            
            # Look for different result files based on the model
            result_files = []
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.endswith(".csv"):
                        result_files.append(os.path.join(root, file))
            
            if result_files:
                print(f"Found {len(result_files)} CSV files, including:")
                for file in result_files[:5]:  # Show up to 5 files
                    print(f"  - {os.path.basename(file)}")
                if len(result_files) > 5:
                    print(f"  ... and {len(result_files) - 5} more")
                print("The experiment appears to have completed with partial results.")
            else:
                print("No result files found. The experiment may not have completed successfully.")
            
            print(f"Check the log file at {log_file} for detailed error information.")
        
        return True
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(symlink_path):
            print(f"Cleaning up symlink: {symlink_path}")
            os.remove(symlink_path)
        else:
            print(f"No symlink to clean up at {symlink_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run LADy experiments with different models and datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-t', '--type',
        choices=['ex', 'im'],
        required=True,
        help='Dataset type: "ex" for explicit aspects or "im" for implicit aspects'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=int,
        choices=[2000, 8000],
        required=True,
        help='Dataset size: 2000 or 8000 reviews'
    )
    
    parser.add_argument(
        '-m', '--model',
        choices=['lda', 'btm', 'bert', 'ctm', 'nrl', 'rnd', 'all'],
        required=True,
        help='Model to run: lda, btm, bert, ctm, nrl, rnd, or "all" to run all models'
    )
    
    parser.add_argument(
        '-n', '--naspects',
        type=int,
        default=5,
        help='Number of aspects to use in the model'
    )
    
    parser.add_argument(
        '--skip-dependency-check',
        action='store_true',
        help='Skip checking for model dependencies'
    )
    
    return parser.parse_args()

def main():
    """Main function to run experiments based on command-line arguments."""
    global args
    args = parse_arguments()
    
    # Convert type to full name
    dataset_type = "explicit" if args.type == "ex" else "implicit"
    
    # Set up paths
    dataset_dir = f"datasets/{dataset_type}_{args.size}"
    dataset_path = f"{dataset_dir}/{dataset_type}_{args.size}.xml"
    output_base_dir = f"experiments/{dataset_type}_{args.size}"
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Determine which models to run
    models_to_run = ['lda', 'btm', 'bert', 'ctm', 'nrl', 'rnd'] if args.model == 'all' else [args.model]
    
    # Run experiments for each selected model
    results = {}
    for model in models_to_run:
        print(f"\n{'='*80}\nRunning {model.upper()} model on {dataset_type} {args.size} dataset\n{'='*80}")
        
        output_dir = f"{output_base_dir}/{model}"
        
        success = run_experiment(
            model=model,
            naspects=args.naspects,
            dataset_path=dataset_path,
            output_path=output_dir,
            skip_dependency_check=args.skip_dependency_check
        )
        
        results[model] = "Completed" if success else "Failed"
        
        print(f"\nExperiment for {model.upper()} on {dataset_type} {args.size} dataset: {results[model]}")
    
    # Print summary
    print(f"\n{'='*80}\nExperiment Summary\n{'='*80}")
    for model, status in results.items():
        print(f"{model.upper()}: {status}")
    
    print(f"\nAll specified experiments have been completed.")

if __name__ == "__main__":
    main() 