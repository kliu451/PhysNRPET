"""
Aleatoric uncertainty for SirenPNR kinetic-parameter estimation.

Two complementary approaches:

Part 1 — Homoscedastic  (no retraining)
    Reads la.sigma_noise from existing Laplace state files.
    sigma_noise is a single global noise scalar fitted during marglik
    optimisation — constant across all voxels.

Part 2 — Heteroscedastic  (requires retraining ~50 epochs)
    Adds a log_var output head to SirenPNR and fine-tunes with Gaussian NLL.
    Produces a per-voxel aleatoric uncertainty map:
        σ_aleatoric(x) = exp(0.5 * log_var(x))

Usage
-----
    # Part 1 — extract sigma_noise from saved Laplace states (no retraining)
    python aleatoric_uncertainty.py --homo

    # Part 2 — fine-tune from model.pt with NLL loss
    python aleatoric_uncertainty.py --hetero --fit

    # Part 2 — eval only from a saved hetero model (no retraining)
    python aleatoric_uncertainty.py --hetero --eval --model hetero_model.pt
"""

import argparse
import datetime
import os
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Net import SirenPNR
from Utils import TAC_2TC_KM

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


def _hms(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m}m {s}s" if m else f"{s}s"


class _Timer:
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        print(f"\n[INFO] [{self.label}] starting …")
        self._t = time.time()
        return self
    def __exit__(self, *_):
        print(f"[INFO] [{self.label}] done in {_hms(time.time() - self._t)}")


# ─────────────────────────────────────────────────────────────────────────────
# BODY MASK  (separate patient from air background)
# ─────────────────────────────────────────────────────────────────────────────

def body_mask(image: np.ndarray, corner_frac: float = 0.05, k_sigma: float = 5.0) -> np.ndarray:
    """
    Segment the patient body from the air background in a 2D PET slice.

    PET scans are centred with air at the edges, so:
      1. Estimate the air noise floor from the image corners (guaranteed air).
      2. Threshold just above it: thr = corner_mean + k_sigma * corner_std.
      3. Flood-fill background INWARD from the border; the body is everything
         NOT connected to the edge. This protects low-uptake *interior* tissue
         (lung, fat) that a plain global threshold would wrongly delete.
      4. Keep the largest connected component and close small gaps.

    Returns a boolean mask (True = body), same (H, W) as `image`.
    Note: Otsu is deliberately avoided — on PET it lands far too high
    (separates hot organs from everything) and discards most of the body.
    """
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=float)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    imn = (img - img.min()) / (img.max() - img.min() + 1e-8)

    H, W = imn.shape
    m = max(4, int(round(min(H, W) * corner_frac)))
    corners = np.concatenate([
        imn[:m, :m].ravel(), imn[:m, -m:].ravel(),
        imn[-m:, :m].ravel(), imn[-m:, -m:].ravel(),
    ])
    thr = float(corners.mean() + k_sigma * corners.std())

    # background = dark region connected to the image border (flood-fill from edges)
    dark = imn <= thr
    lbl, _ = ndi.label(dark)
    border_labels = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border_labels.discard(0)
    background = np.isin(lbl, list(border_labels))
    body = ~background

    # largest connected component, then close small gaps
    lbl2, n2 = ndi.label(body)
    if n2 > 0:
        sizes = np.bincount(lbl2.ravel()); sizes[0] = 0
        body = lbl2 == sizes.argmax()
    body = ndi.binary_closing(body, structure=np.ones((5, 5)))
    print(f"  [DEBUG] body_mask: thr={thr:.5f}  body={100 * body.mean():.1f}% of slice")
    return body


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — HOMOSCEDASTIC
# ─────────────────────────────────────────────────────────────────────────────

def homo_load(laplace_dir: str = "laplace_outputs") -> dict:
    """Extract la.sigma_noise from all saved Laplace state files."""
    results = {}
    print(f"[DEBUG] Searching for Laplace states in: {laplace_dir}")
    for h in tqdm(["diag", "kron", "full"], desc="Loading Laplace states"):
        candidates = sorted(
            f for f in os.listdir(laplace_dir)
            if f.startswith(f"laplace_state_{h}_") and f.endswith(".pt")
        )
        if not candidates:
            print(f"\n  [WARN] no saved state for hessian={h}, skipping")
            continue
        path = os.path.join(laplace_dir, candidates[-1])
        print(f"\n  [INFO] Loading {path} …")
        with tqdm(total=1, desc=f"    torch.load {h}", leave=False):
            la = torch.load(path, map_location="cpu", weights_only=False)
        results[h] = {
            "sigma_noise":     float(la.sigma_noise.item()),
            "prior_precision": float(la.prior_precision.item()),
            "path": path,
        }
        print(f"    [DEBUG] sigma_noise={results[h]['sigma_noise']:.6f}  "
              f"prior_precision={results[h]['prior_precision']:.4f}")
    
    print(f"[INFO] homo_load complete. Found {len(results)} valid states.")
    return results


def homo_visualize(sigma_dict: dict, save_dir: str = "laplace_outputs", image_frame: np.ndarray | None = None, ts: str | None = None):
    ts     = ts or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = list(sigma_dict.keys())
    sigmas = [sigma_dict[h]["sigma_noise"]     for h in labels]
    priors = [sigma_dict[h]["prior_precision"] for h in labels]
    colors = ["#4a90d9", "#e07b39", "#5cb85c"]
    sigma_mean = float(np.mean(sigmas)) if sigmas else 0.0

    print(f"[DEBUG] Starting homo_visualize. labels={labels}, sigma_mean={sigma_mean:.6f}")

    with tqdm(total=1, desc="  Preparing visualization", leave=False):
        pass

    if image_frame is None:
        print("[DEBUG] No image_frame provided to homo_visualize. Plotting stats only.")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            "Homoscedastic Aleatoric Noise — la.sigma_noise\n"
            "(global TAC observation noise std, identical for all voxels)",
            fontsize=11,
        )
    else:
        print(f"[DEBUG] image_frame provided with shape: {image_frame.shape}")
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "Homoscedastic Aleatoric Noise — la.sigma_noise\n"
            "(global TAC observation noise std, identical for all voxels)",
            fontsize=12,
        )
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.25)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, vals, ylabel, title in zip(
        axes,
        [sigmas, priors],
        ["sigma_noise  (TAC noise std)", "prior precision  λ"],
        ["Aleatoric noise level by Hessian", "Prior precision by Hessian"],
    ):
        ax.bar(labels, vals, color=colors[:len(labels)], width=0.5, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.5f}", ha="center", fontsize=9)

    if image_frame is not None:
        img = np.array(image_frame, dtype=float)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = np.rot90(img, k=1)
        ax_img = fig.add_subplot(gs[1, 0])
        ax_overlay = fig.add_subplot(gs[1, 1])
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title("Validation PET image")
        ax_img.axis("off")

        # Constant field → uniform tint + text label, no colorbar
        # (a single global scalar has no spatial variation to decode).
        ax_overlay.imshow(img, cmap="gray", alpha=0.35)
        ax_overlay.imshow(np.ones_like(img), cmap="Purples", alpha=0.30, vmin=0, vmax=1)
        ax_overlay.set_title(
            "Homoscedastic uncertainty overlay\n"
            "(uniform — same σ for every voxel)",
            fontsize=10,
        )
        ax_overlay.axis("off")
        ax_overlay.text(
            0.5, 0.5, f"σ_aleatoric = {sigma_mean:.5f}\n(constant)",
            transform=ax_overlay.transAxes, ha="center", va="center",
            fontsize=12, color="white", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
        )

    plt.tight_layout()
    img_path = os.path.join(save_dir, f"aleatoric_homo_{ts}.png")
    print(f"[INFO] Saving visualization to {img_path}")
    with tqdm(total=1, desc="  Saving PNG", leave=False):
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    lines = [
        "Homoscedastic Aleatoric Uncertainty  (la.sigma_noise)",
        "=" * 62,
        "",
        "sigma_noise is a single scalar fitted during marglik optimisation.",
        "It represents the assumed observation noise std on the TAC values.",
        "Being constant, it tells you the average noise level across all",
        "voxels — not where the data is noisier or cleaner.",
        "",
        f"  {'Hessian':<6}  {'sigma_noise':>12}  {'prior_prec':>12}  source",
        "  " + "-" * 58,
    ]
    for h in labels:
        d = sigma_dict[h]
        lines.append(
            f"  {h:<6}  {d['sigma_noise']:>12.6f}  "
            f"{d['prior_precision']:>12.4f}  {os.path.basename(d['path'])}"
        )
    txt_path = os.path.join(save_dir, f"aleatoric_homo_{ts}_summary.txt")
    print(f"[INFO] Saving summary to {txt_path}")
    with tqdm(total=1, desc="  Saving summary", leave=False):
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — HETEROSCEDASTIC
# ─────────────────────────────────────────────────────────────────────────────

class SirenPNRHetero(SirenPNR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hid = self.k_linear.in_features
        self.log_var_linear = nn.Linear(hid, 1)
        nn.init.zeros_(self.log_var_linear.weight)
        nn.init.constant_(self.log_var_linear.bias, -2.0)  # init: σ ≈ e^{-1} ≈ 0.37

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Use actual parameter device — self.device is unreliable outside a Trainer
        dev = next(self.parameters()).device
        x = x.to(dev)
        if self.B is not None:
            self.B = self.B.to(dev)
            x = torch.matmul(2.0 * torch.pi * x, self.B.T)
            x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.k_linear(self._encode(x)), beta=5)

    def forward_with_log_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xf      = self._encode(x)
        k_hat   = torch.nn.functional.softplus(self.k_linear(xf), beta=5)
        log_var = self.log_var_linear(xf)
        return k_hat, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch
        k_hat, log_var = self.forward_with_log_var(x)

        vb_loss = torch.relu(k_hat[:, 3] - 1.0).mean()

        idif_interp = self.idif_interp.to(self.device).repeat(k_hat.shape[0], 1)
        C_km = TAC_2TC_KM(idif_interp, self.t.to(self.device), k_hat, step=0.03)[:, self.matching_indices]

        mse = ((C_km - y.to(self.device).squeeze(2)) ** 2).mean(dim=1, keepdim=True)
        nll = (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).mean()

        self.log("train_loss",   nll + vb_loss, on_epoch=True)
        self.log("mean_log_var", log_var.mean(), on_epoch=True)
        return nll + vb_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        k_hat, log_var = self.forward_with_log_var(x)

        idif_interp = self.idif_interp.to(self.device).repeat(k_hat.shape[0], 1)
        C_km = TAC_2TC_KM(idif_interp, self.t.to(self.device), k_hat, step=0.03)[:, self.matching_indices]

        # val dataset provides frame 61 only: y is (B, 1); compare against last time point
        y_val = y.to(self.device).view(k_hat.shape[0], 1)
        mse = ((C_km[:, -1:] - y_val) ** 2)
        nll = (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).mean()
        
        self.log("val_loss", nll, on_epoch=True, prog_bar=True)
        self.log("val_mean_log_var", log_var.mean(), on_epoch=True)
        return nll


class HeteroHistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.mean_log_var = []
        self.val_mean_log_var = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        self.epochs.append(epoch)
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss") or metrics.get("train_loss_epoch")
        val_loss = metrics.get("val_loss") or metrics.get("val_loss_epoch")
        mean_log_var = metrics.get("mean_log_var") or metrics.get("mean_log_var_epoch")
        val_mean_log_var = metrics.get("val_mean_log_var") or metrics.get("val_mean_log_var_epoch")
        if train_loss is not None:
            self.train_loss.append(train_loss.item())
        if mean_log_var is not None:
            self.mean_log_var.append(mean_log_var.item())
        if val_loss is not None:
            self.val_loss.append(val_loss.item())
        if val_mean_log_var is not None:
            self.val_mean_log_var.append(val_mean_log_var.item())


def plot_training_history(callback: HeteroHistoryCallback, save_path: str):
    print(f"[INFO] Plotting training history to {save_path}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(callback.epochs, callback.train_loss, label="train_loss", marker="o")
    if callback.val_loss:
        ax.plot(callback.epochs, callback.val_loss, label="val_loss", marker="o")
    ax.set_title("Heteroscedastic Training Progress")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def hetero_fit(
    pretrained_path: str = "model.pt",
    save_path:       str = "hetero_model.pt",
    max_epochs:      int = 50,
    batch_size:      int = 128,
    n_data:          int = 128 * 128,
    z_slice:         int = 230,
):
    from PetDatasets import DynPETQSDataset, Val2DPETDataset

    with _Timer("Load pretrained model"):
        print(f"[DEBUG] Loading weights from {pretrained_path}")
        base = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        base.eval()

    with _Timer("Build SirenPNRHetero"):
        hetero = SirenPNRHetero(
            in_features=base.net[0].linear.in_features,
            hidden_features=base.k_linear.in_features,
            hidden_layers=len(base.net) - 1,
            out_features=base.k_linear.out_features,
            B=base.B,
        )
        missing, unexpected = hetero.load_state_dict(base.state_dict(), strict=False)
        print(f"  [DEBUG] weights copied | missing (new head expected): {missing}")
        print(f"  [DEBUG] unexpected keys: {unexpected}")
        del base

    print(f"\n[INFO] Initializing Datasets for z_slice={z_slice}...")
    train_data   = DynPETQSDataset(sample_size=n_data, z_slice=z_slice)
    val_data     = Val2DPETDataset(z_slice=z_slice, preloaded_tensor=train_data.intensities)
    
    print(f"  [DEBUG] Train dataset size: {len(train_data)}")
    print(f"  [DEBUG] Val dataset size: {len(val_data)}")
    print(f"  [DEBUG] Val spatial shape: {val_data.spatial_shape}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=4096,       num_workers=0)
    hetero.val_spatial_shape = val_data.spatial_shape

    history_cb = HeteroHistoryCallback()
    print("[INFO] Starting PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=100,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[history_cb],
    )
    trainer.fit(hetero, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    plot_path = os.path.join(os.path.dirname(save_path), "hetero_training_progress.png")
    plot_training_history(history_cb, save_path=plot_path)
    
    print(f"[INFO] Saving fine-tuned hetero model to {save_path}")
    torch.save(hetero, save_path)
    return hetero


@torch.no_grad()
def hetero_predict(
    model,
    coords: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    k_list, lv_list = [], []
    n = (len(coords) + batch_size - 1) // batch_size
    print(f"[DEBUG] hetero_predict predicting for {len(coords)} coordinates in {n} batches...")
    
    for start in tqdm(range(0, len(coords), batch_size), total=n, desc="  predict batches"):
        xb = coords[start : start + batch_size]
        k, lv = model.forward_with_log_var(xb)
        k_list.append(k.cpu())
        lv_list.append(lv.cpu())
        
    k_mean          = torch.cat(k_list)
    aleatoric_sigma = torch.exp(0.5 * torch.cat(lv_list))
    print(f"[DEBUG] Predictions complete. Output shapes -> k_mean: {k_mean.shape}, aleatoric_sigma: {aleatoric_sigma.shape}")
    return k_mean, aleatoric_sigma


def hetero_visualize(
    k_mean:          torch.Tensor,
    aleatoric_sigma: torch.Tensor,
    spatial_shape:   tuple,
    save_path:       str = "aleatoric_hetero.png",
    base_image:      np.ndarray | None = None,
    mask:            np.ndarray | None = None,
):
    print(f"[INFO] Starting hetero_visualize for shape {spatial_shape}")
    h, w     = spatial_shape
    k_names  = ["K1", "k2", "k3", "Vb"]

    # Background → NaN so it renders transparent and percentile scaling
    # is computed over body voxels only (not diluted by ~70% air).
    def _apply_mask(arr2d):
        return np.where(mask, arr2d, np.nan) if mask is not None else arr2d

    has_image = base_image is not None
    if has_image:
        print(f"  [DEBUG] Base image provided with shape: {base_image.shape}")
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.9], width_ratios=[1, 1, 1.15],
                      hspace=0.35, wspace=0.25)
    else:
        print("  [DEBUG] No base image provided.")
        fig = plt.figure(figsize=(18, 8))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.15],
                      hspace=0.35, wspace=0.30)

    fig.suptitle(
        "Heteroscedastic Aleatoric Uncertainty\n"
        "MAP kinetic parameters (left)  |  per-voxel σ_aleatoric (right)",
        fontsize=12,
    )

    with tqdm(total=5, desc="  Plotting panels", leave=False) as pbar:
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (r, c), name in zip(positions, k_names):
            ax  = fig.add_subplot(gs[r, c])
            img = np.rot90(_apply_mask(k_mean[:, k_names.index(name)].reshape(h, w).numpy()), k=1)
            vmin, vmax = float(np.nanpercentile(img, 1)), float(np.nanpercentile(img, 99))
            im  = ax.imshow(img, cmap="hot", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}  (MAP)", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            pbar.update(1)

        sigma_img = np.rot90(_apply_mask(aleatoric_sigma.reshape(h, w).numpy()), k=1)
        vmin, vmax = float(np.nanpercentile(sigma_img, 1)), float(np.nanpercentile(sigma_img, 99))
        ax_s = fig.add_subplot(gs[0:2, 2] if has_image else gs[:, 2])
        im_s = ax_s.imshow(sigma_img, cmap="plasma", vmin=vmin, vmax=vmax)
        ax_s.set_title("σ_aleatoric\n(per-voxel TAC noise std)", fontsize=10)
        ax_s.axis("off")
        plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        pbar.update(1)

    if has_image:
        print("[DEBUG] Generating overlays...")
        img = np.array(base_image, dtype=float)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = np.rot90((img - np.nanmin(img)) / ((np.nanmax(img) - np.nanmin(img)) + 1e-8), k=1)
        ax_img = fig.add_subplot(gs[2, 0:2])
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title("Validation PET image")
        ax_img.axis("off")

        ax_overlay = fig.add_subplot(gs[2, 2])
        ax_overlay.imshow(img, cmap="gray", alpha=0.35)
        overlay = sigma_img
        im_ov = ax_overlay.imshow(overlay, cmap="inferno", alpha=0.55, vmin=vmin, vmax=vmax)
        ax_overlay.set_title("Per-voxel σ_aleatoric overlay", fontsize=10)
        ax_overlay.axis("off")
        plt.colorbar(im_ov, ax=ax_overlay, fraction=0.046, pad=0.04)

    print(f"[INFO] Saving composite figure to {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def combined_visualize(
    sigma_dict:      dict,
    aleatoric_sigma: torch.Tensor,
    spatial_shape:   tuple,
    base_image:      np.ndarray,
    save_dir:        str = "laplace_outputs",
    ts:              str | None = None,
    mask:            np.ndarray | None = None,
) -> str:
    """PET | Part 1 (homo) | Part 2 (hetero) | Image-based (Poisson) — all as overlays.

    If `mask` (body mask, True=body) is given, the per-voxel maps (Part 2 and
    Poisson) are restricted to body voxels: background → NaN (transparent), and
    the reported statistics are robust median + IQR over the body only.
    """
    ts  = ts or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = spatial_shape

    sigmas          = [v["sigma_noise"] for v in sigma_dict.values()]
    sigma_homo_mean = float(np.mean(sigmas))
    sigma_homo_std  = float(np.std(sigmas)) if len(sigmas) > 1 else 0.0

    # ---- unrotated arrays (for masking + stats) ----
    sig_hw = aleatoric_sigma.reshape(h, w).numpy()
    img_hw = np.array(base_image, dtype=float)
    if img_hw.ndim == 3 and img_hw.shape[-1] == 1:
        img_hw = img_hw[..., 0]
    img_hw = (img_hw - np.nanmin(img_hw)) / ((np.nanmax(img_hw) - np.nanmin(img_hw)) + 1e-8)
    poisson_hw = np.sqrt(np.clip(img_hw, 0, None))

    if mask is None:
        mask = np.ones(sig_hw.shape, dtype=bool)
    body_pct = 100 * mask.mean()

    def _stats(a):                       # robust summary over body voxels
        b = a[mask]
        return np.median(b), np.percentile(b, 25), np.percentile(b, 75)
    sh_med, sh_q1, sh_q3 = _stats(sig_hw)
    pj_med, pj_q1, pj_q3 = _stats(poisson_hw)

    def _disp(a):                        # mask → NaN, then rotate upright for display
        return np.rot90(np.where(mask, a, np.nan), k=1)
    img           = np.rot90(img_hw, k=1)          # PET panel: show full image
    sigma_hetero  = _disp(sig_hw)
    poisson_sigma = _disp(poisson_hw)
    mask_rot      = np.rot90(mask, k=1)

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    fig.suptitle(
        "Aleatoric Uncertainty  —  Part 1 (homoscedastic)  |  Part 2 (heteroscedastic)  |  Image-based (Poisson)"
        f"   [body-masked, {body_pct:.0f}% of slice]",
        fontsize=13,
    )

    vmin2 = float(np.nanpercentile(sigma_hetero, 1))
    vmax2 = float(np.nanpercentile(sigma_hetero, 99))
    vp1   = float(np.nanpercentile(poisson_sigma, 1))
    vp99  = float(np.nanpercentile(poisson_sigma, 99))
    std_label = f" ± {sigma_homo_std:.5f}" if sigma_homo_std > 0 else ""

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("PET image  (frame 61, z=230)")
    axes[0].axis("off")

    # Part 1 is a single global scalar → uniform tint over the body + text label.
    # No colorbar: a constant field has no spatial variation to decode.
    axes[1].imshow(img, cmap="gray", alpha=0.5)
    axes[1].imshow(np.where(mask_rot, 1.0, np.nan), cmap="Purples", alpha=0.35, vmin=0, vmax=1)
    axes[1].set_title(
        f"Part 1 — Homoscedastic\nσ = {sigma_homo_mean:.5f}{std_label}\n(constant across all voxels)"
    )
    axes[1].axis("off")
    axes[1].text(
        0.5, 0.5, f"σ = {sigma_homo_mean:.5f}\n(uniform)",
        transform=axes[1].transAxes, ha="center", va="center",
        fontsize=13, color="white", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.55),
    )

    axes[2].imshow(img, cmap="gray", alpha=0.35)
    im2 = axes[2].imshow(sigma_hetero, cmap="inferno", alpha=0.6, vmin=vmin2, vmax=vmax2)
    axes[2].set_title(
        f"Part 2 — Heteroscedastic\nmedian = {sh_med:.5f}  IQR [{sh_q1:.5f}, {sh_q3:.5f}]\n(per-voxel, body)"
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="σ_aleatoric")

    axes[3].imshow(img, cmap="gray", alpha=0.4)
    im3 = axes[3].imshow(poisson_sigma, cmap="plasma", alpha=0.6, vmin=vp1, vmax=vp99)
    axes[3].set_title(
        f"Image-based — Poisson σ ∝ √I\nmedian = {pj_med:.5f}  IQR [{pj_q1:.5f}, {pj_q3:.5f}]\n(no model, body)"
    )
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="σ_poisson")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"aleatoric_combined_{ts}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Combined visualization saved to {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — IMAGE-BASED NOISE (no model)
# ─────────────────────────────────────────────────────────────────────────────

def image_noise_visualize(
    image: np.ndarray,
    save_dir: str = "laplace_outputs",
    local_patch: int = 5,
) -> str:
    """
    Estimate aleatoric noise purely from the PET image, no model required.

    Two estimates:
      Poisson  — σ(x) ∝ √I(x)  (PET counts are Poisson; variance = mean)
      Local σ  — std of pixel intensities in a (local_patch × local_patch) window
    """
    from scipy.ndimage import generic_filter

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img = np.array(image, dtype=float)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    img_norm = np.rot90((img - np.nanmin(img)) / ((np.nanmax(img) - np.nanmin(img)) + 1e-8), k=1)

    poisson_sigma = np.sqrt(np.clip(img_norm, 0, None))

    local_sigma = generic_filter(img_norm, np.std, size=local_patch)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "Image-based Aleatoric Noise Estimates  (no model)\n"
        "Left: PET image  |  Centre: Poisson σ ∝ √I  |  Right: local patch σ",
        fontsize=12,
    )

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("PET image  (frame 61, z=230)")
    axes[0].axis("off")

    axes[1].imshow(img_norm, cmap="gray", alpha=0.4)
    vp1, vp99 = np.percentile(poisson_sigma, 1), np.percentile(poisson_sigma, 99)
    im1 = axes[1].imshow(poisson_sigma, cmap="plasma", alpha=0.6, vmin=vp1, vmax=vp99)
    axes[1].set_title(
        f"Poisson noise  σ ∝ √I\n"
        f"σ̄ = {poisson_sigma.mean():.4f} ± {poisson_sigma.std():.4f}"
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="σ_poisson")

    axes[2].imshow(img_norm, cmap="gray", alpha=0.4)
    vl1, vl99 = np.percentile(local_sigma, 1), np.percentile(local_sigma, 99)
    im2 = axes[2].imshow(local_sigma, cmap="plasma", alpha=0.6, vmin=vl1, vmax=vl99)
    axes[2].set_title(
        f"Local patch σ  ({local_patch}×{local_patch} window)\n"
        f"σ̄ = {local_sigma.mean():.4f} ± {local_sigma.std():.4f}"
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="σ_local")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"aleatoric_image_{ts}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Image-based noise map saved to {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Aleatoric uncertainty for SirenPNR",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Part 1 — homoscedastic (reads saved Laplace states)\n"
            "  python aleatoric_uncertainty.py --homo\n\n"
            "  # Part 2 — fine-tune from model.pt\n"
            "  python aleatoric_uncertainty.py --hetero --fit\n\n"
            "  # Part 2 — eval only (uses hetero_model.pt by default)\n"
            "  python aleatoric_uncertainty.py --hetero --eval\n\n"
            "  # Both parts + combined viz (no retraining)\n"
            "  python aleatoric_uncertainty.py --homo --hetero --eval\n"
        ),
    )
    p.add_argument("--homo",       action="store_true", help="Run Part 1: homoscedastic sigma_noise")
    p.add_argument("--hetero",     action="store_true", help="Run Part 2: heteroscedastic log_var head")
    p.add_argument("--fit",        action="store_true", help="Fine-tune SirenPNRHetero (Part 2 only)")
    p.add_argument("--eval",       action="store_true", help="Evaluate saved hetero model (Part 2 only)")
    p.add_argument("--model",      default=None,
                   help="Model file. For --fit: pretrained SirenPNR (default: model.pt). "
                        "For --eval: saved hetero model (default: hetero_model.pt).")
    p.add_argument("--save_model", default="hetero_model.pt", help="Where to save the trained hetero model")
    p.add_argument("--output_dir",  default="aleatoric_outputs", help="Directory for aleatoric output files")
    p.add_argument("--laplace_dir", default="laplace_outputs",   help="Directory to read saved Laplace states from (Part 1)")
    p.add_argument("--z_slice",    type=int, default=230,      help="Axial slice index")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_data",     type=int, default=128 * 128)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Run started at {ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.homo and not args.hetero:
        print("[WARN] Neither --homo nor --hetero specified. Exiting.")
        p.print_help()
        return

    do_fit  = args.hetero and (args.fit  or not args.eval)
    do_eval = args.hetero and (args.eval or not args.fit)

    # ── Load val dataset once (shared between homo viz and hetero eval) ──────
    val_data   = None
    base_image = None
    if args.homo or do_eval:
        from PetDatasets import Val2DPETDataset
        with _Timer("Load Val2DPETDataset"):
            val_data = Val2DPETDataset(z_slice=args.z_slice)
        base_image = val_data.intensities.squeeze(-1).numpy()
        print(f"  [DEBUG] base_image shape: {base_image.shape}")

    sigma_dict    = {}
    aleatoric_sigma = None
    k_mean          = None

    # ── Part 1: Homoscedastic ──────────────────────────────────────────────
    if args.homo:
        print("\n" + "=" * 60)
        print("  PART 1 — HOMOSCEDASTIC ALEATORIC UNCERTAINTY")
        print("=" * 60)
        sigma_dict = homo_load(args.laplace_dir)
        if sigma_dict:
            homo_visualize(sigma_dict, save_dir=args.output_dir, image_frame=base_image, ts=ts)
        else:
            print("[WARN] No Laplace states found. Cannot run Part 1 visualization.")

    # ── Part 2: Heteroscedastic ────────────────────────────────────────────
    if args.hetero:
        print("\n" + "=" * 60)
        print("  PART 2 — HETEROSCEDASTIC ALEATORIC UNCERTAINTY")
        print("=" * 60)
        print(f"[DEBUG] Flags -> fit: {do_fit}, eval: {do_eval}")

        # Resolve model path
        if args.model is not None:
            model_path = args.model
        elif do_eval and not do_fit:
            model_path = "hetero_model.pt"
            print(f"[INFO] Auto-selected model: {model_path}")
        else:
            model_path = "model.pt"

        if do_fit:
            model = hetero_fit(
                pretrained_path=model_path,
                save_path=args.save_model,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                n_data=args.n_data,
                z_slice=args.z_slice,
            )
        else:
            with _Timer("Load hetero model"):
                print(f"[INFO] Loading saved hetero model from {model_path}...")
                model = torch.load(model_path, map_location=device, weights_only=False)
                model.eval()

        if do_eval:
            print(f"\n[INFO] Starting Evaluation phase...")

            # Use the already-loaded val_data (avoids a second 28 GB load)
            if val_data is None:
                from PetDatasets import Val2DPETDataset
                with _Timer("Initialize Val2DPETDataset"):
                    val_data = Val2DPETDataset(z_slice=args.z_slice)
                base_image = val_data.intensities.squeeze(-1).numpy()

            print(f"  [DEBUG] Evaluation dataset length: {len(val_data)}")
            coords = torch.stack([val_data[i][0] for i in tqdm(range(len(val_data)), desc="  Loading eval coords")])
            print(f"  [DEBUG] Eval coords shape: {coords.shape}")

            k_mean, aleatoric_sigma = hetero_predict(model, coords)

            k_path    = os.path.join(args.output_dir, f"hetero_k_mean_{ts}.pt")
            s_path    = os.path.join(args.output_dir, f"hetero_aleatoric_sigma_{ts}.pt")
            np_k_path = os.path.join(args.output_dir, f"hetero_k_mean_{ts}.npy")
            np_s_path = os.path.join(args.output_dir, f"hetero_aleatoric_sigma_{ts}.npy")
            print("[INFO] Saving raw tensor outputs to disk...")
            torch.save(k_mean, k_path)
            torch.save(aleatoric_sigma, s_path)
            np.save(np_k_path, k_mean.numpy())
            np.save(np_s_path, aleatoric_sigma.numpy())

            # Body mask (separate patient from air) — built from the displayed slice.
            print("[INFO] Building body mask (corner noise floor + flood-fill)...")
            mask2d = body_mask(base_image)
            hh, ww = val_data.spatial_shape

            plot_path = os.path.join(args.output_dir, f"aleatoric_hetero_{ts}.png")
            hetero_visualize(k_mean, aleatoric_sigma, val_data.spatial_shape,
                             save_path=plot_path, base_image=base_image, mask=mask2d)

            print("\n[INFO] Compiling Summary Statistics...")
            k_names = ["K1", "k2", "k3", "Vb"]
            lines = [
                "Heteroscedastic Aleatoric Uncertainty Summary",
                "=" * 62,
                "",
                "MAP kinetic parameters (across all voxels):",
                f"  {'Param':<5}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}",
                "  " + "-" * 44,
            ]
            for i, name in enumerate(k_names):
                v = k_mean[:, i]
                lines.append(f"  {name:<5}  {v.mean():>8.4f}  {v.std():>8.4f}  "
                             f"{v.min():>8.4f}  {v.max():>8.4f}")
            lines += [
                "",
                "Per-voxel aleatoric sigma  exp(0.5 * log_var):",
                f"  mean={aleatoric_sigma.mean():.5f}  std={aleatoric_sigma.std():.5f}  "
                f"min={aleatoric_sigma.min():.5f}  max={aleatoric_sigma.max():.5f}",
                "",
                f"Body-masked (flood-fill, {100 * mask2d.mean():.1f}% of slice) — robust stats:",
                "  (median + IQR over body voxels only; background air excluded)",
            ]
            sig_body = aleatoric_sigma.reshape(hh, ww).numpy()[mask2d]
            lines.append(
                f"  sigma   median={np.median(sig_body):8.5f}  "
                f"IQR=[{np.percentile(sig_body, 25):.5f}, {np.percentile(sig_body, 75):.5f}]  "
                f"mean={sig_body.mean():.5f}"
            )
            for i, name in enumerate(k_names):
                vb = k_mean[:, i].reshape(hh, ww).numpy()[mask2d]
                lines.append(
                    f"  {name:<5}   median={np.median(vb):8.4f}  "
                    f"IQR=[{np.percentile(vb, 25):.4f}, {np.percentile(vb, 75):.4f}]"
                )
            txt_path = os.path.join(args.output_dir, f"aleatoric_hetero_{ts}_summary.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[INFO] Summary saved to {txt_path}")

    # ── Combined visualization when both parts ran ─────────────────────────
    if args.homo and do_eval and sigma_dict and aleatoric_sigma is not None:
        combined_visualize(
            sigma_dict, aleatoric_sigma,
            val_data.spatial_shape, base_image,
            save_dir=args.output_dir, ts=ts, mask=mask2d,
        )


if __name__ == "__main__":
    main()