#this script will be run in a conda Env that is already confirmed to satisfy app requirements (voxaboxen).
#look up documentation and code for app here: https://github.com/earthspecies/voxaboxen

import os
import sys
import shutil
import zipfile
import tarfile
import subprocess
import yaml
from pathlib import Path
sys.path.append(os.getcwd())
from user.misc import arg_loader

#check GPU abailable
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Device Count: {torch.cuda.device_count()}')

args=arg_loader()
#below is as follows. CWD is app CWD which is at: INSTINCT/bin
#../cache/399753/538819/307679/919206 ../cache/399753/538819/307679/919206/545214  simple-v1-0

# Define Input: args[1] is the dir, append the filename expected from R
INPUT_DIR = Path(args[1]).resolve()
MANIFESTS_ZIP = INPUT_DIR / 'VoxaboxenManifests.zip'

# Define Output: args[2] is the dir, append target filename
OUTPUT_DIR = Path(args[2]).resolve()
OUTPUT_TARBALL = OUTPUT_DIR / "Voxaboxen_results.tar.gz"

# --- 2. CHANGE WORKING DIRECTORY ---
# Switch to the output cache directory so tmp folders are created there
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

# Define Tmp Dirs (Now relative to OUTPUT_DIR because of os.chdir)
DATA_DIR_TMP = Path("./data_dir_tmp").resolve()
PROJECT_DIR_TMP = Path("./local_tmp").resolve()

# --- 3. HARDCODED APP PATHS ---
VOXABOXEN_REPO = Path("C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen")
BEATS_CHECKPOINT = Path("C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen/weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")

def run_command(cmd_list, working_dir=None):
    """
    Executes a shell command. 
    stdout/stderr are NOT captured, allowing them to flow to the console
    so you can see progress bars and logs in real-time.
    """
    print(f"\n[Wrapper] Executing: {' '.join(cmd_list)}")
    try:
        # check=True raises CalledProcessError if exit code != 0
        subprocess.run(cmd_list, cwd=working_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Wrapper] Error: Command failed with exit code {e.returncode}")
        sys.exit(1)
        
def main():
    # Resolve paths
    vox_repo_path = VOXABOXEN_REPO.resolve()
    beats_path = BEATS_CHECKPOINT.resolve()

    # Clean Start
    if DATA_DIR_TMP.exists(): shutil.rmtree(DATA_DIR_TMP)
    if PROJECT_DIR_TMP.exists(): shutil.rmtree(PROJECT_DIR_TMP)
    
    DATA_DIR_TMP.mkdir(parents=True)
    PROJECT_DIR_TMP.parent.mkdir(parents=True, exist_ok=True)

    try:


        # 1. Run Project Setup
        # FIX: Use .as_posix() to enforce forward slashes (matches working manual snippet).
        # FIX: Explicitly pass manifest paths to prevent validation crash in params.py.
        # FIX: Use --arg=val syntax as requested.
        setup_cmd = [
            sys.executable, "main.py", "project-setup",
            f"--data-dir={DATA_DIR_TMP.as_posix()}",
            f"--project-dir={PROJECT_DIR_TMP.as_posix()}",
            f"--train-info-fp={(DATA_DIR_TMP / 'train_info.csv').as_posix()}",
            f"--val-info-fp={(DATA_DIR_TMP / 'val_info.csv').as_posix()}",
            f"--test-info-fp={(DATA_DIR_TMP / 'test_info.csv').as_posix()}"
        ]
        run_command(setup_cmd, working_dir=vox_repo_path)

        # 2. Extract Manifests
        print(f"[Wrapper] Extracting {MANIFESTS_ZIP}...")
        if not MANIFESTS_ZIP.exists():
            print(f"[Wrapper] Error: {MANIFESTS_ZIP} not found.")
            sys.exit(1)

        with zipfile.ZipFile(MANIFESTS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR_TMP)

        # 3. Modify Configuration
        config_fp = PROJECT_DIR_TMP / "project_config.yaml"
        print(f"[Wrapper] Updating configuration at {config_fp}...")
        
        with open(config_fp, 'r') as f:
            config_data = yaml.safe_load(f)

        # Enforce 'positive' class
        config_data['label_set'] = ['positive']
        config_data['unknown_label'] = "out_of_effort"
        
        config_data['label_mapping'] = {"out_of_effort":"out_of_effort","positive":"positive"}
        
        with open(config_fp, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # 4. Run Training Pipeline
        #128 batch size fits in n1. Trying 192 - that was too big. After encoder unfreezes, memory explosion. For first attempt, keep this fixed
        train_cmd = [
            sys.executable, "main.py", "train-model",
            f"--project-config-fp={config_fp.as_posix()}",
            "--name=test",
            "--n-epochs=3",
            "--batch-size=200",
            "--encoder-type=beats",
            f"--beats-checkpoint-fp={beats_path.as_posix()}",
            #"--gradient-accumulation-steps=8" #try this with v low batch size if allowing encoder to unfreeze
            #"--bidirectional",
            #"--early-stopping",
            #"--val-during-training"
            
        ]
        run_command(train_cmd, working_dir=vox_repo_path)

        # 5. Bundle Outputs
        print(f"[Wrapper] Bundling results into {OUTPUT_TARBALL}...")
        with tarfile.open(OUTPUT_TARBALL, "w:gz") as tar:
            # We are in OUTPUT_DIR (from os.chdir), so just add the folder name
            tar.add(PROJECT_DIR_TMP.name, arcname="voxaboxen_project_output")

        print("[Wrapper] Pipeline complete.")

    except Exception as e:
        print(f"\n[Wrapper] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    #finally:
    #    # --- CLEANUP ---
    #    print("\n[Wrapper] Cleaning up temporary directories...")
    #    if DATA_DIR_TMP.exists(): shutil.rmtree(DATA_DIR_TMP)
    #    if PROJECT_DIR_TMP.exists(): shutil.rmtree(PROJECT_DIR_TMP)
    #    print("[Wrapper] Cleanup done.")

if __name__ == "__main__":
    main()
#expand zips into 'data_dir_tmp'

#environment config:
#run in CMD. 
#python main.py project-setup --data-dir=data_dir_tmp --project-dir="./local_tmp"

#Go into exported config file generated from last cmd. Need to make sure that the label 'positive' is what we are looking for and the only label specified. 

#run the training pipeline:

#python main.py train-model --project-config-fp="project_config.yaml" --name=demo --n-epochs=50 --batch-size=4 --encoder-type=beats --beats-checkpoint-fp=weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional

#allow stdout to flow normally to wrapper process, allowing for early stopping etc. 

#output: bundle all created files into a tarball. 

