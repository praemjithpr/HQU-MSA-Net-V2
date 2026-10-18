import os
import sys
import glob
import time
import numpy as np
import torch
import nibabel as nib

def prepare_all_1000_patients():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.getenv("BRATS_DATA_DIR", os.path.join(base_dir, "data", "BraTS-MEN-Train"))
    cache_dir = os.path.join(base_dir, "processed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"[*] Raw BraTS dataset directory '{data_dir}' not found.")
        print(f"[*] Populating demo cache files in '{cache_dir}'...")
        from fast_dataset import FastCachedBraTSDataset
        dataset = FastCachedBraTSDataset(cache_dir=cache_dir)
        return

    patient_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    
    print("=" * 70)
    print(f"[*] Pre-processing ALL {len(patient_dirs)} Patients into `.pt` Tensors...")
    print("=" * 70)
    
    count = 0
    skipped = 0
    start = time.time()
    
    for idx, p_dir in enumerate(patient_dirs, 1):
        p_id = os.path.basename(p_dir)
        save_check = os.path.join(cache_dir, f"{p_id}_slice_45.pt")
        
        if os.path.exists(save_check):
            skipped += 1
            continue
            
        t1c_path = os.path.join(p_dir, f"{p_id}-t1c.nii.gz")
        t1n_path = os.path.join(p_dir, f"{p_id}-t1n.nii.gz")
        t2w_path = os.path.join(p_dir, f"{p_id}-t2w.nii.gz")
        t2f_path = os.path.join(p_dir, f"{p_id}-t2f.nii.gz")
        seg_path = os.path.join(p_dir, f"{p_id}-seg.nii.gz")
        
        if not (os.path.exists(t1c_path) and os.path.exists(seg_path)):
            continue
            
        try:
            t1c_vol = nib.load(t1c_path).get_fdata(dtype=np.float32)
            t1n_vol = nib.load(t1n_path).get_fdata(dtype=np.float32)
            t2w_vol = nib.load(t2w_path).get_fdata(dtype=np.float32)
            t2f_vol = nib.load(t2f_path).get_fdata(dtype=np.float32)
            seg_vol = nib.load(seg_path).get_fdata(dtype=np.float32)
            
            for slice_idx in range(45, 115, 4):
                seg_slice = seg_vol[:, :, slice_idx]
                mask = (seg_slice > 0).astype(np.float32)
                
                def norm(img):
                    m = img > 0
                    return (img - img[m].mean()) / (img[m].std() + 1e-8) if m.any() else img
                    
                t1c_s = norm(t1c_vol[:, :, slice_idx])
                t1n_s = norm(t1n_vol[:, :, slice_idx])
                t2w_s = norm(t2w_vol[:, :, slice_idx])
                t2f_s = norm(t2f_vol[:, :, slice_idx])
                
                img_tensor = torch.from_numpy(np.stack([t1c_s, t1n_s, t2w_s, t2f_s], axis=0))
                mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0))
                
                img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)
                
                save_file = os.path.join(cache_dir, f"{p_id}_slice_{slice_idx}.pt")
                torch.save({'image': img_tensor, 'mask': mask_tensor, 'pid': p_id}, save_file)
                count += 1
                
        except Exception as e:
            continue
            
        if idx % 50 == 0 or idx == len(patient_dirs):
            pct = (idx / len(patient_dirs)) * 100
            print(f" -> [{idx:04d}/{len(patient_dirs):04d}] ({pct:5.1f}%) | Preprocessed Slices: {count} | Cached: {skipped}")
            
    print(f"[+] Complete! Cached {count} new tensor slices in {time.time() - start:.1f}s.")

if __name__ == "__main__":
    prepare_all_1000_patients()
