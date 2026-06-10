import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import glob
from pytorch_lightning.loggers import WandbLogger
import wandb
from Net import SirenPNR
from PetDatasets import DynPETQSDataset, Val2DPETDataset
torch.autograd.set_detect_anomaly(False)
torch.set_float32_matmul_precision('medium')
torch.manual_seed(seed=0)
torch.cuda.manual_seed(seed=0)
torch.mps.manual_seed(seed=0)

import multiprocessing as mp
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

def preflight_check():
    data_dir = "Uncertainty_Eval/20_pat_25"
    pet_files = glob.glob(os.path.join(data_dir, "PET_*.nii.gz"))
    errors = []
    if not os.path.isdir(data_dir):
        errors.append(f"Data directory not found: {os.path.abspath(data_dir)}")
    elif len(pet_files) == 0:
        errors.append(f"No PET_*.nii.gz files found in {os.path.abspath(data_dir)}")
    else:
        print(f"[OK] Found {len(pet_files)} PET files in {os.path.abspath(data_dir)}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise SystemExit("Preflight check failed. Aborting.")

def main_inr():
    preflight_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    train_data = DynPETQSDataset(sample_size=128*128)
    print("len train: ", len(train_data))
    val_data = Val2DPETDataset()
    train_loader = DataLoader(train_data, batch_size=128, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=4096, num_workers=0)
    wandb_logger = WandbLogger(project="PhysNRPET")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=10, save_top_k=-1)

    mapping_size = 256
    B_gauss = torch.randn((mapping_size, 3)).to(device) #coords
    B_gausshu = torch.randn((mapping_size, 4)).to(device) #coords + hu
    model = SirenPNR(in_features=mapping_size*2, B=B_gauss*10, out_features=4,
                hidden_layers=3)
    model.val_spatial_shape = val_data.spatial_shape
    # model =SirenPNR(in_features=mapping_size*2, B=B_gausshu*10, out_features=4,
    #             hidden_layers=3)
    # model =SirenPNR(in_features=mapping_size*2+4096, B=B_gauss*10, out_features=4,
    #             hidden_layers=3)

    trainer = pl.Trainer(max_epochs=101,
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_cb],
                        check_val_every_n_epoch=10,
                        gradient_clip_val=100,
                        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(model, 'model.pt')

if __name__ == "__main__":
    main_inr()