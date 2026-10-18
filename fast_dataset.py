import os
import glob
import torch
from torch.utils.data import Dataset

class FastCachedBraTSDataset(Dataset):
    """
    Ultra-Fast In-Memory / Pre-cached PyTorch Dataset.
    Loads pre-processed `.pt` tensor files instantly into RAM / GPU VRAM.
    """
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if len(self.cache_files) == 0:
            print(f"[*] Cache directory '{cache_dir}' is empty. Generating synthetic sample slice tensors for demonstration...")
            self._generate_synthetic_cache(cache_dir, num_patients=10, slices_per_patient=5)
            self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    def _generate_synthetic_cache(self, cache_dir, num_patients=10, slices_per_patient=5):
        for p_idx in range(1, num_patients + 1):
            p_id = f"BraTS-MEN-{p_idx:05d}-000"
            for slice_idx in range(50, 50 + slices_per_patient * 10, 10):
                # Create synthetic 4-channel MRI (128x128)
                torch.manual_seed(p_idx * 100 + slice_idx)
                img = torch.randn(4, 128, 128)
                
                # Synthetic tumor mask at center
                mask = torch.zeros(1, 128, 128)
                y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
                center_y = 50 + (p_idx % 5) * 6
                center_x = 50 + (slice_idx % 4) * 6
                radius = 15 + (p_idx % 3) * 3
                dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
                mask[0, dist < radius] = 1.0
                
                # Add tumor signal enhancement to image channels
                img[:, dist < radius] += 1.5
                
                save_file = os.path.join(cache_dir, f"{p_id}_slice_{slice_idx}.pt")
                torch.save({'image': img, 'mask': mask, 'pid': p_id}, save_file)
        print(f"[+] Generated {num_patients * slices_per_patient} sample tensor slices in '{cache_dir}'")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_files[idx], weights_only=True)
        return data['image'], data['mask'], data['pid']

