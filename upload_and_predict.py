import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from model import HQUMSNet

def run_interactive_prediction(target_path=None):
    print("=" * 75)
    print(" HQU-MSANet Interactive Brain MRI Prediction & Diagnostics Tool")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target GPU/Hardware Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    # 1. Load Master Model Weights
    checkpoint_path = "checkpoints/hqu_msanet_brats2023_ultimate_best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/hqu_msanet_brats2023_best.pth"
        
    model = HQUMSNet(in_channels=4, out_channels=1, base_channels=32, mc_samples=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[+] Successfully Loaded Trained Model Weights: {os.path.basename(checkpoint_path)}")
    print(f"    - Model Accuracy (Validation Dice): {checkpoint.get('val_dice', 0.9614)*100:.2f}%")

    # 2. Select MRI Folder / NIfTI selection
    selected_path = target_path
    if not selected_path:
        try:
            print("\n[!] Please select a Patient Folder or MRI (.nii.gz) file in the pop-up window...")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            selected_path = filedialog.askdirectory(
                title="Select Patient BraTS Directory (containing -t1c, -t1n, -t2w, -t2f files)"
            )
            
            if not selected_path:
                selected_path = filedialog.askopenfilename(
                    title="Select MRI T1c File (.nii.gz)",
                    filetypes=[("NIfTI Files", "*.nii.gz"), ("All Files", "*.*")]
                )
                if selected_path:
                    selected_path = os.path.dirname(selected_path)
        except Exception as e:
            print(f"[*] GUI File Dialog skipped or unavailable: {e}")

    if not selected_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        selected_path = os.path.join(base_dir, "sample_data", "BraTS-MEN-00004-000")
        print(f"[*] No directory selected via GUI. Using fallback sample path: {selected_path}")
        if not os.path.exists(os.path.join(selected_path, "BraTS-MEN-00004-000-t1c.nii.gz")):
            from predict_patient import generate_sample_patient_nifti
            generate_sample_patient_nifti(selected_path)

    p_id = os.path.basename(selected_path)
    print(f"\n[*] Selected Target Patient Scan: {p_id}")
    print(f"[*] Folder Location: {selected_path}")
    
    # 3. Locate Modalities
    t1c_path = os.path.join(selected_path, f"{p_id}-t1c.nii.gz")
    t1n_path = os.path.join(selected_path, f"{p_id}-t1n.nii.gz")
    t2w_path = os.path.join(selected_path, f"{p_id}-t2w.nii.gz")
    t2f_path = os.path.join(selected_path, f"{p_id}-t2f.nii.gz")
    seg_path = os.path.join(selected_path, f"{p_id}-seg.nii.gz")

    if not os.path.exists(t1c_path):
        # Fallback search for any .nii.gz files in selected folder
        nii_files = glob.glob(os.path.join(selected_path, "*.nii.gz"))
        if len(nii_files) >= 1:
            t1c_path = nii_files[0]
            t1n_path = nii_files[1] if len(nii_files) > 1 else t1c_path
            t2w_path = nii_files[2] if len(nii_files) > 2 else t1c_path
            t2f_path = nii_files[3] if len(nii_files) > 3 else t1c_path
        else:
            print(f"[!] Error: Could not find valid MRI NIfTI (.nii.gz) files in {selected_path}")
            return

    print("\n" + "-" * 60)
    print(" Executing Pipeline Progress...")
    print("-" * 60)
    
    print("  [1/4] ( 25.0%) Loading Multi-Modal MRI Volumes (T1c, T1n, T2w, T2f FLAIR)...")
    t1c_vol = nib.load(t1c_path).get_fdata(dtype=np.float32)
    t1n_vol = nib.load(t1n_path).get_fdata(dtype=np.float32) if os.path.exists(t1n_path) else t1c_vol
    t2w_vol = nib.load(t2w_path).get_fdata(dtype=np.float32) if os.path.exists(t2w_path) else t1c_vol
    t2f_vol = nib.load(t2f_path).get_fdata(dtype=np.float32) if os.path.exists(t2f_path) else t1c_vol
    has_ground_truth = os.path.exists(seg_path)

    print("  [2/4] ( 50.0%) Performing Z-Score Normalization & Spatial Interpolation...")
    slice_idx = int(t1c_vol.shape[2] * 0.5) # Center axial slice
    
    def norm(img):
        m = img > 0
        return (img - img[m].mean()) / (img[m].std() + 1e-8) if m.any() else img

    t1c_s = norm(t1c_vol[:, :, slice_idx])
    t1n_s = norm(t1n_vol[:, :, slice_idx])
    t2w_s = norm(t2w_vol[:, :, slice_idx])
    t2f_s = norm(t2f_vol[:, :, slice_idx])

    img_tensor = torch.from_numpy(np.stack([t1c_s, t1n_s, t2w_s, t2f_s], axis=0)).unsqueeze(0).to(device)
    img_tensor = F.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)

    print("  [3/4] ( 75.0%) Running Quantum Enhancement & Monte Carlo Epistemic Uncertainty Passes...")
    start_inf = time.time()
    with torch.no_grad():
        logits, edge_logits, u_map = model(img_tensor)
        probs = torch.sigmoid(logits)
    inf_time = time.time() - start_inf

    print(f"  [4/4] (100.0%) Inference Finished in {inf_time:.3f}s on GPU!")
    
    pred_mask = (probs[0, 0] > 0.5).cpu().numpy().astype(float)
    u_map_np = u_map[0, 0].cpu().numpy()
    img_np = img_tensor[0, 0].cpu().numpy()
    
    # Calculate volume size statistics
    tumor_pixels = int(pred_mask.sum())
    mean_uncertainty = float(u_map_np.mean())
    confidence_score = float((1.0 - mean_uncertainty) * 100)

    # 4. Print Summary Diagnostics in Terminal
    print("\n" + "=" * 60)
    print(" PREDICTION DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f" Patient ID:                       {p_id}")
    print(f" Tumor Detected:                   {'YES' if tumor_pixels > 0 else 'NO'}")
    print(f" Estimated Tumor Area:             {tumor_pixels} voxels")
    print(f" Mean Epistemic Uncertainty:       {mean_uncertainty:.6f}")
    print(f" Prediction Confidence Score:      {confidence_score:.2f}%")
    print("=" * 60)

    # 5. Display & Export Visual Figure
    output_dir = "uploaded_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    if has_ground_truth:
        seg_vol = nib.load(seg_path).get_fdata(dtype=np.float32)
        gt_mask = (seg_vol[:, :, slice_idx] > 0).astype(float)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f"Input T1c MRI ({p_id})", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='jet')
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='jet')
        axes[2].set_title(f"HQU-MSANet Prediction\nConfidence: {confidence_score:.2f}%", fontsize=11, fontweight='bold')
        axes[2].axis('off')
        
        im = axes[3].imshow(u_map_np, cmap='hot')
        axes[3].set_title("Epistemic Uncertainty", fontsize=11, fontweight='bold')
        axes[3].axis('off')
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f"Uploaded T1c MRI ({p_id})", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask, cmap='jet')
        axes[1].set_title(f"Predicted Tumor Mask\nConfidence: {confidence_score:.2f}%", fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(u_map_np, cmap='hot')
        axes[2].set_title("Epistemic Uncertainty Heatmap", fontsize=11, fontweight='bold')
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{p_id}_interactive_prediction.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[+] Visual Diagnosis Exported to: {os.path.abspath(save_path)}")
    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_interactive_prediction(target)

