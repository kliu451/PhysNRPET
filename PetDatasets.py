import os
import glob
import numpy as np
import torch
import time
from torch.utils.data import Dataset
import monai
from monai.transforms import LoadImaged, ScaleIntensityRanged, ToTensord
from natsort import natsorted
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DynPETQSDataset(Dataset):
    def __init__(self, sample_size=2, z_slice=230):
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.idx = 0

        data_dir = "Uncertainty_Eval/20_pat_25"
        print(f"[INFO] Scanning directory: {data_dir}")
        image_tensor = natsorted(glob.glob(os.path.join(data_dir, "PET_*")))
        image_dict = [
            {"image": image_name} for image_name in image_tensor
        ]

        transforms = monai.transforms.Compose([
            LoadImaged(keys=["image"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=200000, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["image"])
        ])

        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"liverslice_z{z_slice}.pt")
        legacy_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liverslice09.pt")
        
        if os.path.exists(cache_path):
            file_size_gb = os.path.getsize(cache_path) / (1024**3)
            print(f"[INFO] Loading cached tensor ({file_size_gb:.2f} GB) from {cache_path}... (Please wait)")
            t0 = time.time()
            image_tensor = torch.load(cache_path, weights_only=False)
            print(f"[INFO] Successfully loaded PET data in {time.time() - t0:.1f} seconds.")
        elif z_slice == 230 and os.path.exists(legacy_cache):
            file_size_gb = os.path.getsize(legacy_cache) / (1024**3)
            print(f"[INFO] Loading legacy cached tensor ({file_size_gb:.2f} GB) from {legacy_cache}... (Please wait)")
            t0 = time.time()
            image_tensor = torch.load(legacy_cache, weights_only=False)
            print(f"[INFO] Successfully loaded PET data in {time.time() - t0:.1f} seconds.")
            # Save a compact slice cache so future runs skip the 28 GB load.
            print(f"[INFO] Saving compact z-slice cache to {cache_path}...")
            torch.save(image_tensor.contiguous(), cache_path)
            print(f"[INFO] Compact cache saved ({os.path.getsize(cache_path)/(1024**3):.2f} GB).")
        else:
            dynpetdata = [transforms(d) for d in tqdm(image_dict, desc="Loading raw PET NIfTI frames")]
            print("[INFO] Stacking 4D image tensor... this may require high RAM.")
            image_tensor = torch.stack(
                [entry['image'][:, z_slice:z_slice+1, :] for entry in dynpetdata],
                dim=0,
            )
            print(f"[INFO] Saving cache to {cache_path}...")
            torch.save(image_tensor, cache_path)

        T, D, H, W = image_tensor.shape
        print(f"[INFO] DynPETQSDataset Image Tensor Shape: {image_tensor.shape} | N images: {T}")
        self.spatial_shape = (D, W)
        
        print("[INFO] Generating spatial coordinates for DynPETQSDataset...")
        with tqdm(total=3, desc="Processing Coordinates", leave=False) as pbar:
            self.coords = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'), -1).reshape(-1, 3).float()
            pbar.update(1)
            
            # Normalize coordinates to [-1, 1]
            self.coords = self.coords / torch.tensor([D/2, H/2, W/2]) - 1
            pbar.update(1)
            
            # Flatten the image volume to match coordinates
            self.intensities = image_tensor
            self.sample_size = sample_size
            pbar.update(1)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, _):
        idx = torch.randint(0, len(self.coords), (1,)).item() # uniform sampling
        intensities = self.intensities.reshape(self.intensities.shape[0], -1)[:, idx].reshape(-1, 1)
        return self.coords[idx], intensities

class Val2DPETDataset(Dataset):
    def __init__(self, z_slice=230, preloaded_tensor=None):
        torch.manual_seed(0)
        np.random.seed(0)
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"liverslice_z{z_slice}.pt")
        legacy_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liverslice09.pt")

        print(f"\n[INFO] Initializing Val2DPETDataset for z_slice={z_slice}...")
        if preloaded_tensor is not None:
            print(f"[INFO] Reusing preloaded tensor of shape {preloaded_tensor.shape} (no second disk load).")
            image = preloaded_tensor
            own_image = False
        elif os.path.exists(cache_path):
            file_size_gb = os.path.getsize(cache_path) / (1024**3)
            print(f"[INFO] Loading cached tensor ({file_size_gb:.2f} GB) from {cache_path}... (Please wait)")
            t0 = time.time()
            image = torch.load(cache_path, weights_only=False)
            print(f"[INFO] Load complete in {time.time() - t0:.1f} seconds.")
            own_image = True
        elif z_slice == 230 and os.path.exists(legacy_cache):
            file_size_gb = os.path.getsize(legacy_cache) / (1024**3)
            print(f"[INFO] Loading legacy cached tensor ({file_size_gb:.2f} GB) from {legacy_cache}... (Please wait)")
            t0 = time.time()
            image = torch.load(legacy_cache, weights_only=False)
            print(f"[INFO] Load complete in {time.time() - t0:.1f} seconds.")
            own_image = True
            if not os.path.exists(cache_path):
                print(f"[INFO] Saving compact z-slice cache to {cache_path}...")
                torch.save(image.contiguous(), cache_path)
                print(f"[INFO] Compact cache saved ({os.path.getsize(cache_path)/(1024**3):.2f} GB).")
        else:
            raise FileNotFoundError(
                f"Cached slice file not found for z={z_slice}. Expected {cache_path}"
            )

        print("[INFO] Validation tensor loaded. Cloning frame 61...")
        img = image[61].clone() #pick a frame to validate
        if own_image:
            del image
        
        D, H, W = img.shape
        self.spatial_shape = (D, W)
        
        print("[INFO] Generating spatial coordinates for Val2DPETDataset...")
        with tqdm(total=3, desc="Processing Val Coordinates", leave=False) as pbar:
            self.coords = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'), -1)
            pbar.update(1)
            
            # Normalize coordinates to [-1, 1]
            self.coords = self.coords[:,0,:].unsqueeze(2).reshape(-1, 3).float()
            pbar.update(1)
            
            self.coords = self.coords / torch.tensor([D/2, H/2, W/2]) - 1
            self.intensities = img[:,0,:].unsqueeze(2)
            del img
            pbar.update(1)

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, index):
        intensity = self.intensities.reshape(-1, 1)[index]
        return self.coords[index], torch.tensor([intensity])