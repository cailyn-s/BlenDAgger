import h5py
import os
import argparse
import shutil
import numpy as np

def patch_pretraining_dataset(src_path, target_folder):
    """Copy the pretraining dataset and add human_actions/alphas with the shapes training expects."""
    if not src_path or not os.path.exists(src_path):
        print(f"Source pretraining path {src_path} not found. Skipping.")
        return

    file_name = os.path.basename(src_path)
    dest_path = os.path.join(target_folder, f"patched_{file_name}")

    print(f"Copying pretraining dataset to: {dest_path}")
    shutil.copy2(src_path, dest_path)

    with h5py.File(dest_path, "r+") as f:
        for d in f["data"].keys():
            demo_grp = f[f"data/{d}"]
            
            # Fix human_actions (Always match actions shape, e.g., 12)
            if "human_actions" not in demo_grp:
                demo_grp.create_dataset("human_actions", data=demo_grp["actions"][:])
            
            # Fix alphas (Always strictly 6)
            alpha_dim = 6
            if "alphas" not in demo_grp:
                demo_grp.create_dataset("alphas", data=np.zeros((demo_grp["actions"].shape[0], alpha_dim)))
            elif demo_grp["alphas"].shape[1] != alpha_dim:
                # If it's 12 or something else, slice it down to 6
                existing_alphas = demo_grp["alphas"][:, :alpha_dim]
                del demo_grp["alphas"]
                demo_grp.create_dataset("alphas", data=existing_alphas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Output folder for the patched copy (patched_<name>.hdf5)")
    parser.add_argument("--pretrain_path", type=str, default=None, help="Source pretraining dataset path")
    args = parser.parse_args()

    if args.pretrain_path:
        patch_pretraining_dataset(args.pretrain_path, args.folder)

    print("\nBackfill complete.")

if __name__ == "__main__":
    main()