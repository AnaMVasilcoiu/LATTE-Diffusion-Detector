import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

def compute_folder_dz_maps(folder: str, target_label: int, device="cpu"):
    found = False
    latents = []
    for file in sorted(Path(folder).rglob("*.pt")):
        if not found:
            data = torch.load(file, map_location=device)
            for sample in data.values():
                if sample.get("label", -1) == target_label:
                    latents.append(sample["latent_seq"].to(device))
                    found = True

    if not latents:
        return None, 0

    latents = torch.stack(latents, dim=0)  # [N, T, C, H, W]
    N, T, C, H, W = latents.shape
    maps_sum = [torch.zeros(H, W, device=device) for _ in range(1, T)]

    for t in range(1, T):
        dz = latents[:, t] - latents[:, t - 1]   # [N, C, H, W]
        norm = dz.norm(dim=1)                    # [N, H, W]
        maps_sum[t - 1] = norm.sum(0)            # [H, W]

    return maps_sum, N

def accumulate_dz_maps(folders: list[str], target_label: int, device="cpu"):
    total_maps = None
    total_samples = 0

    for folder in folders:
        print(f"  → processing {folder}")
        maps_sum, n = compute_folder_dz_maps(folder, target_label, device)
        if maps_sum is None or n == 0:
            continue
        if total_maps is None:
            total_maps = [m.clone() for m in maps_sum]
        else:
            for i in range(len(total_maps)):
                total_maps[i] += maps_sum[i]
        total_samples += n

    if total_samples == 0:
        return []

    mean_maps = [m / total_samples for m in total_maps]
    return [m.cpu().numpy() for m in mean_maps]

def plot_all(real_maps, fake_maps_list, labels, outpath: Path, timesteps: list[int]):
    # 1) flatten all real + fake maps into one list of arrays
    all_maps = real_maps + [m for (fmaps, _) in fake_maps_list for m in fmaps]

    # 2) compute global vmin, vmax
    vmin = min(m.min() for m in all_maps)
    vmax = max(m.max() for m in all_maps)

    n_fake = len(fake_maps_list)
    Tm1   = len(real_maps)
    n_rows = 1 + n_fake

    fig, axes = plt.subplots(n_rows, Tm1,
                             figsize=(2.4*Tm1, 2.4*n_rows),
                             gridspec_kw={"wspace":0.05,"hspace":0.05})
    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05)

    # plot real row
    for i in range(Tm1):
        ax = axes[0, i]
        im = ax.imshow(real_maps[i], cmap="hot", vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel("Real", weight="bold")
        ax.set_title(f"{timesteps[i]}→{timesteps[i+1]}")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # plot fake rows
    for r, (fmaps, label) in enumerate(fake_maps_list, start=1):
        for i in range(Tm1):
            ax = axes[r, i]
            ax.imshow(fmaps[i], cmap="hot", vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_ylabel(label, weight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

    # colorbar
    cax = fig.add_axes([0.94, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cax)

    fig.tight_layout(rect=[0,0,0.94,1])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved figure to:", outpath)

def main():
    parser = argparse.ArgumentParser("Δz heatmaps: 1 real row + N fake rows")
    parser.add_argument("--real_dirs", nargs="+", required=True, help="Folders with label=0 samples")
    parser.add_argument("--fake_dirs", nargs="+", required=True, help="Folders with label=1 samples")
    parser.add_argument("--labels", nargs="+", required=True, help="One label per fake_dir (e.g. ADM BigGAN …)")
    parser.add_argument("--out", type=str, default="heatmaps/all_models.png", help="Output PNG path")
    parser.add_argument("--device", type=str, default="cpu", help="torch device: cpu or cuda")
    parser.add_argument("--tracked_timesteps", type=lambda x: list(map(int, x.strip("[]").split(","))), default=[981,741,521,261,1])
    args = parser.parse_args()

    assert len(args.fake_dirs) == len(args.labels), \
        "Must supply exactly one label per fake_dir"

    print("Computing REAL maps …")
    real_maps = accumulate_dz_maps(args.real_dirs, target_label=0, device=args.device)
    if not real_maps:
        raise RuntimeError("No real maps found; check --real_dirs")

    print("Computing FAKE maps …")
    fake_maps_list = []
    for folder, label in zip(args.fake_dirs, args.labels):
        fmaps = accumulate_dz_maps([folder], target_label=1, device=args.device)
        if not fmaps:
            print(f"  !! warning: no data in {folder}, skipping")
            continue
        fake_maps_list.append((fmaps, label))

    if not fake_maps_list:
        raise RuntimeError("No fake maps found; check --fake_dirs")

    # final plot
    plot_all(real_maps, fake_maps_list, args.labels,
             Path(args.out), args.tracked_timesteps)

if __name__ == "__main__":
    main()
