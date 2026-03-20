"""
Generate Rate-Distortion-Perception curves: Baseline (HFLIC) vs Ours (DISTS+LPIPS).
Style matches HFLIC paper Figure 3. Evaluated on Kodak dataset.

Each subplot selects the best epoch for that metric within each lambda run:
- PSNR: max
- MS-SSIM: max
- LPIPS: min
"""
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "baseline": {
        "label": "HFLIC (Baseline)",
        "color": "#1565C0",
        "marker": "s",
        "logs": [
            (1.0, ROOT / "experiments/hflic_stage2_baseline-1.0-bpp/val_hflic_stage2_baseline_260310-120736.log"),
            (0.6, ROOT / "experiments/hflic_stage2_baseline-06-bpp/val_hflic_stage2_baseline_260309-131602.log"),
            (0.3, ROOT / "experiments/hflic_stage2_baseline-03-bpp/val_hflic_stage2_baseline_260308-133559.log"),
        ],
    },
    "ours": {
        "label": "Ours (DISTS+LPIPS)",
        "color": "#C62828",
        "marker": "o",
        "logs": [
            (1.0, ROOT / "experiments/hflic_stage2_ours_dists_lpips-1.0-bpp/val_hflic_stage2_ours_dists_lpips_260315-233131.log"),
            (0.6, ROOT / "experiments/hflic_stage2_ours_dists_lpips-0.6-bpp/val_hflic_stage2_ours_dists_lpips_260314-224130.log"),
            (0.3, ROOT / "experiments/hflic_stage2_ours_dists_lpips-0.3-bpp/val_hflic_stage2_ours_dists_lpips_260313-184833.log"),
        ],
    },
}

LINE_RE = re.compile(
    r"Test epoch (?P<epoch>\d+): .*?"
    r"LPIPS(?: loss)?: (?P<lpips>-?\d+(?:\.\d+)?) .*?"
    r"Bpp(?: rate)?(?: loss)?: (?P<bpp>-?\d+(?:\.\d+)?) .*?"
    r"PSNR: (?P<psnr>-?\d+(?:\.\d+)?) dB \| "
    r"MS-SSIM: (?P<msssim>-?\d+(?:\.\d+)?) dB"
)


def parse_log(log_path):
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            entries.append(
                {
                    "epoch": int(match.group("epoch")),
                    "bpp": float(match.group("bpp")),
                    "psnr": float(match.group("psnr")),
                    "msssim": float(match.group("msssim")),
                    "lpips": float(match.group("lpips")),
                }
            )
    if not entries:
        raise ValueError(f"No validation entries parsed from {log_path}")
    return entries


def select_best_entry(entries, metric_key, lower_better=False):
    if lower_better:
        return min(entries, key=lambda x: (x[metric_key], x["bpp"], -x["epoch"]))
    return max(entries, key=lambda x: (x[metric_key], -x["bpp"], -x["epoch"]))


def build_series(metric_key, lower_better=False):
    series = []
    for method in RUNS.values():
        points = []
        for lambda_bpp, log_path in method["logs"]:
            entry = select_best_entry(parse_log(log_path), metric_key, lower_better=lower_better)
            points.append(
                {
                    "lambda_bpp": lambda_bpp,
                    "epoch": entry["epoch"],
                    "bpp": entry["bpp"],
                    "value": entry[metric_key],
                }
            )
        points.sort(key=lambda x: x["bpp"])
        series.append(
            {
                "label": method["label"],
                "color": method["color"],
                "marker": method["marker"],
                "points": points,
            }
        )
    return series

# ============================================================
# Plotting
# ============================================================

def plot_rd(ax, metric_key, ylabel, lower_better=False):
    for m in build_series(metric_key, lower_better=lower_better):
        bpp = np.array([p["bpp"] for p in m["points"]])
        val = np.array([p["value"] for p in m["points"]])
        ax.plot(
            bpp,
            val,
            color=m["color"],
            marker=m["marker"],
            linestyle="-",
            markersize=7,
            linewidth=2.0,
            label=m["label"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
    ax.set_xlabel('Bit-rate (bpp)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.92, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3, linewidth=0.5)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

plot_rd(axes[0], 'psnr',   'PSNR (dB)')
plot_rd(axes[1], 'msssim', 'MS-SSIM (dB)')
plot_rd(axes[2], 'lpips',  'LPIPS', lower_better=True)

fig.suptitle('Rate-Distortion-Perception on Kodak',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout(w_pad=3.0)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'thesis-template-LS4CV2', 'figures')
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, 'rd_curves.pdf')
png_path = os.path.join(out_dir, 'rd_curves.png')

fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
fig.savefig(png_path, bbox_inches='tight', dpi=200)

print(f"Saved: {os.path.abspath(pdf_path)}")
print(f"Saved: {os.path.abspath(png_path)}")

for metric_key, lower_better in [("psnr", False), ("msssim", False), ("lpips", True)]:
    print("\n" + "=" * 92)
    print(f"{metric_key.upper()}-optimal points")
    print(f"{'Method':<22} {'λ_bpp':>6} {'Epoch':>6} {'BPP':>8} {metric_key.upper():>10}")
    print("-" * 92)
    for m in build_series(metric_key, lower_better=lower_better):
        for p in m["points"]:
            print(
                f"{m['label']:<22} {p['lambda_bpp']:>6.1f} {p['epoch']:>6d} "
                f"{p['bpp']:>8.4f} {p['value']:>10.4f}"
            )
    print("=" * 92)
