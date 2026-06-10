"""
Post-hoc Laplace approximation for SirenPNR kinetic-parameter uncertainty.
No retraining needed — wraps the pretrained model.pt.

Install dependency first:
    pip install laplace-torch

Usage:
    python laplace_uncertainty.py            # fit Laplace + visualise
    python laplace_uncertainty.py --fit      # fit only  → laplace_state.pt
    python laplace_uncertainty.py --eval     # load saved state, visualise

Design note:
    We use TACWrapper (coords → predicted TAC) so the Hessian is computed
    through the full 2TC kinetic model, consistent with the training loss.
    The last nn.Linear (siren.k_linear: 512→4) is auto-detected by laplace-torch.
    K-parameter uncertainty is recovered via MC sampling from that posterior.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils import torch_conv_batch


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


class _Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        print(f"\n[{self.label}] starting …")
        self._t = time.time()
        return self

    def __exit__(self, *_):
        print(f"[{self.label}] done in {_hms(time.time() - self._t)}")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


def _tac_forward(siren, k_hat: torch.Tensor) -> torch.Tensor:
    """
    Inline 2TC kinetic model — identical to TAC_2TC_KM but without the
    `TAC.requires_grad_()` call that would detach the autograd graph needed
    for the Laplace Hessian.
    """
    idif = siren.idif_interp.repeat(k_hat.shape[0], 1)
    k1, k2, k3, Vb = k_hat.T.unbind(0)
    k1 = k1.unsqueeze(1).unsqueeze(2)
    k2 = k2.unsqueeze(1).unsqueeze(2)
    k3 = k3.unsqueeze(1).unsqueeze(2)
    Vb = Vb.unsqueeze(1)
    t  = siren.t.repeat(k1.shape[0], 1, 1)
    a  = idif.unsqueeze(0)
    e  = (k2 + k3) * t
    b  = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
    c  = torch_conv_batch(a, b) * 0.03
    TAC = (1 - Vb) * c.squeeze(0) + Vb * a.squeeze(0)
    return TAC[:, siren.matching_indices]           # (B, T_measured)


class TACWrapper(nn.Module):
    """coords → predicted TAC at measurement times.

    The Laplace Hessian is computed through the full physics chain so
    uncertainty is consistent with the MSE-TAC training objective.
    """
    def __init__(self, siren):
        super().__init__()
        self.siren = siren

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_hat = self.siren(x)
        return _tac_forward(self.siren, k_hat)


class _TqdmLoader:
    """DataLoader wrapper that adds a tqdm progress bar while preserving
    .dataset and other attributes laplace-torch accesses internally."""
    def __init__(self, loader: DataLoader, **tqdm_kwargs):
        self._loader = loader
        self._tqdm_kwargs = tqdm_kwargs
        self.dataset = loader.dataset

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        return iter(tqdm(self._loader, **self._tqdm_kwargs))


def _laplace_loader(dataset, siren, batch_size: int = 512, fit_device: str = "cpu") -> DataLoader:
    """Returns (coords, k_param_pseudo_labels) batches.

    Using k-param pseudo-labels (siren_MAP output) makes k_linear the true
    output layer so laplace-torch's last_layer_jacobians works correctly for
    all three hessian structures (diag, kron, full).
    """
    def collate(batch):
        xs, _ = zip(*batch)
        xs = torch.stack(xs).to(fit_device)
        with torch.no_grad():
            ys = siren(xs)   # (B, 4) k-param pseudo-labels
        return xs, ys
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, collate_fn=collate)


def fit_laplace(
    model_path: str = "model.pt",
    save_path:  str = "laplace_state.pt",
    hessian:    str = "kron",
    n_data:     int = 128 * 128,
    n_steps:    int = 50,
    batch_size: int = 512,
):
    from laplace import Laplace
    from PetDatasets import DynPETQSDataset

    fit_device = "cpu"

    with _Timer("Load model"):
        siren = torch.load(model_path, map_location=fit_device, weights_only=False)
        siren.eval()
        print(f"  last layer: k_linear {tuple(siren.k_linear.weight.shape)} "
              f"({siren.k_linear.weight.numel() + siren.k_linear.bias.numel()} params)")

    with _Timer("Load training data"):
        dataset = DynPETQSDataset(sample_size=n_data)
        loader  = _laplace_loader(dataset, siren=siren, fit_device=fit_device, batch_size=batch_size)
        print(f"  {n_data} samples | {len(loader)} batches of {batch_size}")

    la = Laplace(siren, likelihood="regression",
                 subset_of_weights="last_layer",
                 hessian_structure=hessian)

    with _Timer("GGN Hessian"):
        la.fit(_TqdmLoader(loader, desc="  batches", total=len(loader)))

    with _Timer("Prior precision (marglik)"):
        la.optimize_prior_precision(method="marglik", n_steps=n_steps, lr=0.1,
                                    verbose=True, progress_bar=True)
        print(f"  prior precision = {la.prior_precision.item():.4f}")
        print(f"  sigma noise     = {la.sigma_noise.item():.4f}")

    with _Timer("Save"):
        for m in la.model.modules():
            m._forward_hooks.clear()
        torch.save(la, save_path)
        import os
        print(f"  {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")

    return la, siren


def load_laplace(state_path: str = "laplace_state.pt"):
    with _Timer("Load Laplace state"):
        la = torch.load(state_path, map_location="cpu", weights_only=False)
        siren = la.model.model  # FeatureExtractor wraps siren directly
        siren.eval()
        print(f"  loaded from {state_path}")
    return la, siren


@torch.no_grad()
def predict_k_uncertainty(
    la,
    siren,
    coords:    torch.Tensor,
    n_samples: int = 200,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample the Laplace posterior over k_linear weights to get k-param uncertainty.

    Returns
    -------
    k_mean : (N, 4)  MAP kinetic parameters (K1, k2, k3, Vb)
    k_std  : (N, 4)  posterior std of kinetic parameters
    """
    siren_device = next(siren.parameters()).device
    coords   = coords.to(siren_device)
    k_linear = siren.k_linear
    n_out, n_in = k_linear.weight.shape
    print(f"\n  {len(coords)} voxels | {n_samples} posterior samples | device: {siren_device}")

    with _Timer("Draw posterior samples"):
        samples = la.sample(n_samples=n_samples)
        print(f"  sample shape: {tuple(samples.shape)}")

    with _Timer("Extract SIREN features"):
        feats_list = []
        n_batches = (len(coords) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(coords), batch_size), total=n_batches, desc="  feature batches"):
            xb = coords[start:start + batch_size]
            if siren.B is not None:
                xf = torch.matmul(2.0 * torch.pi * xb, siren.B.T)
                xf = torch.cat([torch.sin(xf), torch.cos(xf)], -1)
            else:
                xf = xb
            feats_list.append(siren.net(xf))
        feats = torch.cat(feats_list)
        print(f"  features: {tuple(feats.shape)}")

    k_mean = torch.nn.functional.softplus(
        torch.nn.functional.linear(feats, k_linear.weight, k_linear.bias), beta=5
    )
    print(f"  MAP k-params range  K1=[{k_mean[:,0].min():.3f}, {k_mean[:,0].max():.3f}]  "
          f"Vb=[{k_mean[:,3].min():.3f}, {k_mean[:,3].max():.3f}]")

    with _Timer("MC uncertainty sampling"):
        k_var = torch.zeros_like(k_mean)
        for s in tqdm(samples, desc="  samples", total=n_samples):
            w  = s[:n_in * n_out].reshape(n_out, n_in)
            b  = s[n_in * n_out:]
            ki = torch.nn.functional.softplus(
                torch.nn.functional.linear(feats, w, b), beta=5
            )
            k_var += (ki - k_mean).pow(2)

    k_std = (k_var / n_samples).sqrt()

    k_names = ["K1", "k2", "k3", "Vb"]
    lines = [
        "┌─────────────────────────────────────────────────────────────┐",
        "│        Kinetic Parameter Summary (across all voxels)        │",
        "├──────┬──────────────────────────┬──────────────────────────┤",
        "│ Param│    MAP estimate           │  Laplace uncertainty     │",
        "│      │  mean ± std  [min, max]   │  mean std  [min, max]    │",
        "├──────┼──────────────────────────┼──────────────────────────┤",
    ]
    for i, name in enumerate(k_names):
        m = k_mean[:, i]
        s = k_std[:, i]
        lines.append(
            f"│ {name:<4s} │ {m.mean():.4f} ± {m.std():.4f} "
            f"[{m.min():.4f}, {m.max():.4f}] │ "
            f"{s.mean():.4f}      [{s.min():.4f}, {s.max():.4f}] │"
        )
    lines.append("└──────┴──────────────────────────┴──────────────────────────┘")
    summary = "\n".join(lines)
    print("\n" + summary)

    return k_mean, k_std, summary


def visualize(
    k_mean: torch.Tensor,
    k_std:  torch.Tensor,
    spatial_shape: tuple,
    save_path: str = "laplace_uncertainty.png",
):
    h, w = spatial_shape
    k_names = ["K1", "k2", "k3", "Vb"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Kinetic Parameters — MAP Estimate & Laplace Posterior Std", fontsize=13)

    for i, name in enumerate(k_names):
        ki_map = k_mean[:, i].reshape(h, w).numpy()
        ki_std = k_std[:, i].reshape(h, w).numpy()

        im0 = axes[0, i].imshow(ki_map, cmap="hot")
        axes[0, i].set_title(f"{name} (MAP)")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        im1 = axes[1, i].imshow(ki_std, cmap="viridis")
        axes[1, i].set_title(f"{name} Std (Laplace)")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def visualize_combined(
    k_mean:    torch.Tensor,
    std_dict:  dict,            # {"diag": tensor, "kron": tensor, "full": tensor}
    spatial_shape: tuple,
    save_path: str = "laplace_combined.png",
):
    """4-row × 4-col grid: MAP estimate + one row per Hessian approximation."""
    h, w = spatial_shape
    k_names    = ["K1", "k2", "k3", "Vb"]
    row_labels = ["MAP estimate"] + [f"{k.capitalize()} Hessian σ" for k in std_dict]
    all_data   = [k_mean] + list(std_dict.values())
    cmaps      = ["hot"] + ["viridis"] * len(std_dict)

    n_rows, n_cols = len(all_data), len(k_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5 + 0.6))
    fig.suptitle(
        "Kinetic Parameters — MAP Estimate vs. Laplace Posterior σ\n"
        "(Diagonal / Kronecker-factored / Full Hessian approximations)",
        fontsize=13,
    )

    for r, (data, cmap, row_label) in enumerate(zip(all_data, cmaps, row_labels)):
        for c, name in enumerate(k_names):
            ax  = axes[r, c]
            img = data[:, c].reshape(h, w).cpu().numpy()
            vmin = float(np.percentile(img, 1))
            vmax = float(np.percentile(img, 99))
            im  = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=10, pad=3)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold",
                              color="#222222", labelpad=6)
                ax.axis("on")
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def main():
    import datetime
    p = argparse.ArgumentParser(
        description="Laplace uncertainty for SirenPNR",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single hessian — fit + eval (first run)\n"
            "  python laplace_uncertainty.py --fit --eval --hessian kron\n\n"
            "  # Single hessian — eval only using saved state (NO retraining)\n"
            "  python laplace_uncertainty.py --eval --state laplace_outputs/laplace_state_kron_<ts>.pt\n\n"
            "  # Combined 3-hessian image from saved states (NO retraining)\n"
            "  python laplace_uncertainty.py --compare \\\n"
            "      --state_diag laplace_outputs/laplace_state_diag_<ts>.pt \\\n"
            "      --state_kron laplace_outputs/laplace_state_kron_<ts>.pt \\\n"
            "      --state_full laplace_outputs/laplace_state_full_<ts>.pt\n"
        ),
    )
    p.add_argument("--fit",        action="store_true", help="Fit Laplace (required first time)")
    p.add_argument("--eval",       action="store_true", help="Evaluate / visualise (single hessian)")
    p.add_argument("--compare",    action="store_true",
                   help="Load all 3 hessian states and produce a combined 4-row image")
    p.add_argument("--test",       action="store_true", help="Tiny run to verify no crashes")
    p.add_argument("--model",      default="model.pt")
    p.add_argument("--state",      default=None, help="State file path (single-hessian mode)")
    p.add_argument("--state_diag", default=None, help="Diag state .pt  (--compare mode)")
    p.add_argument("--state_kron", default=None, help="Kron state .pt  (--compare mode)")
    p.add_argument("--state_full", default=None, help="Full state .pt  (--compare mode)")
    p.add_argument("--hessian",    default="kron", choices=["diag", "kron", "full"],
                   help="kron=fast+good, diag=fastest, full=most accurate but slow")
    p.add_argument("--n_samples",  type=int, default=200,
                   help="Posterior samples for k-param uncertainty")
    p.add_argument("--output_dir", default="laplace_outputs",
                   help="Directory to store state and plot files")
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test:
        args.fit = True
        args.n_samples = 5

    # ── COMPARE MODE: all 3 hessians → one combined image ─────────────────────
    if args.compare:
        from PetDatasets import Val2DPETDataset

        hessians    = ["diag", "kron", "full"]
        state_paths = {
            "diag": args.state_diag,
            "kron": args.state_kron,
            "full": args.state_full,
        }

        # Auto-detect latest saved state for any unspecified hessian
        for h in hessians:
            if state_paths[h] is None:
                candidates = sorted(
                    f for f in os.listdir(args.output_dir)
                    if f.startswith(f"laplace_state_{h}_") and f.endswith(".pt")
                )
                if candidates:
                    state_paths[h] = os.path.join(args.output_dir, candidates[-1])
                    print(f"  auto-detected {h} state: {state_paths[h]}")
                else:
                    raise FileNotFoundError(
                        f"No saved state for hessian='{h}' found in {args.output_dir}. "
                        f"Run with --fit --hessian {h} first, or pass --state_{h} <path>."
                    )

        # Load val coords once
        val_data = Val2DPETDataset()
        coords   = torch.stack([val_data[i][0] for i in tqdm(range(len(val_data)), desc="  loading coords")])

        std_results = {}
        k_map_ref   = None

        all_summaries = []
        for h in hessians:
            print(f"\n{'='*60}\n  Hessian: {h.upper()}  —  {state_paths[h]}\n{'='*60}")
            la, siren = load_laplace(state_paths[h])
            k_mean, k_std, summary = predict_k_uncertainty(la, siren, coords, args.n_samples)
            std_results[h] = k_std
            if k_map_ref is None:
                k_map_ref = k_mean
            all_summaries.append(f"{'='*65}\nhessian: {h}\n{'='*65}\n\n{summary}")

        combined_path = os.path.join(args.output_dir, f"laplace_combined_{ts}.png")
        visualize_combined(k_map_ref, std_results, val_data.spatial_shape, save_path=combined_path)

        table_path = os.path.join(args.output_dir, f"laplace_combined_{ts}_summary.txt")
        with open(table_path, "w") as f:
            f.write("\n\n".join(all_summaries) + "\n")
        print(f"Saved {table_path}")
        return

    # ── SINGLE HESSIAN MODE ───────────────────────────────────────────────────
    if args.state is None:
        args.state = os.path.join(args.output_dir, f"laplace_state_{args.hessian}_{ts}.pt")

    plot_path = os.path.join(args.output_dir, f"laplace_uncertainty_{args.hessian}_{ts}.png")

    do_fit  = args.fit  or not args.eval
    do_eval = args.eval or not args.fit

    fit_kwargs = dict(n_data=256, n_steps=3, batch_size=128) if args.test else {}
    la, siren = (
        fit_laplace(args.model, args.state, args.hessian, **fit_kwargs)
        if do_fit else
        load_laplace(args.state)
    )

    if do_eval:
        from PetDatasets import Val2DPETDataset
        val_data = Val2DPETDataset()
        coords   = torch.stack([val_data[i][0] for i in tqdm(range(len(val_data)), desc="  loading coords")])

        print(f"MC sampling with {args.n_samples} posterior samples …")
        k_mean, k_std, summary = predict_k_uncertainty(la, siren, coords, args.n_samples)
        visualize(k_mean, k_std, val_data.spatial_shape, save_path=plot_path)
        table_path = plot_path.replace(".png", "_summary.txt")
        with open(table_path, "w") as f:
            f.write(f"hessian: {args.hessian}\n\n{summary}\n")
        print(f"Saved {table_path}")


if __name__ == "__main__":
    main()