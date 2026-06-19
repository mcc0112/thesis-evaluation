#!/usr/bin/env python3
"""
plot_avg_lbd.py
---------------
Plots median "Avg LBD" against configuration (n, p), grouped by
propagator variant (V0-V3), with IQR error bars.

Follows the same style conventions (fonts, colors, layout helpers) as
generate_figures.py.

Expected input CSV format (long/wide hybrid, one row per metric per
configuration):

    metric,config_n,config_p,config,V0_median,V0_iqr,V0_n,
    V1_median,V1_iqr,V1_n,V2_median,V2_iqr,V2_n,V3_median,V3_iqr,V3_n

Only rows where `metric == <--metric>` (default: "Avg LBD") are used.

Note on error bars: the input file provides a single IQR value per
variant/config (not separate Q1/Q3). Error bars are therefore drawn
symmetrically as median +/- IQR/2. If your CSV instead has separate
Q1/Q3 columns, adapt the `get_yerr` function below.

-------------------------------------------------------------------
USAGE
-------------------------------------------------------------------
    python plot_avg_lbd.py --csv path/to/your_data.csv

Optional arguments:
    --metric   Name of the metric row to plot (default: "Avg LBD")
    --out      Output PDF path (default: figures/avg_lbd_by_config.pdf)

Example:
    python plot_avg_lbd.py --csv summary_table.csv --metric "Avg LBD" \
        --out figures/F_avg_lbd.pdf
-------------------------------------------------------------------
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Shared style (matches generate_figures.py)
# ------------------------------------------------------------------

VARIANTS = ["V0", "V1", "V2", "V3"]

VARIANT_LABELS = {
    "V0": "V0 (decomposed)",
    "V1": "V1 (conflict-only)",
    "V2": "V2 (full GAC)",
    "V3": "V3 (Hall-circuit)",
}

VARIANT_COLORS = {
    "V0": "#4C72B0",
    "V1": "#DD8452",
    "V2": "#C44E52",
    "V3": "#55A868",
}

# Publication-ready defaults
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "figure.dpi":        300,
})


def config_label(n, p):
    return f"n={int(n)}, p={p / 100:.2f}"


def save_pdf(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_offsets(n_variants):
    width = 0.8 / n_variants
    offsets = (
        np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants)
        * width
    )
    return width, offsets


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

def plot_metric_by_config(df: pd.DataFrame, metric: str, out_path: Path):
    """
    Grouped bar chart of median metric value per configuration, one
    group of bars (V0-V3) per configuration, with IQR error bars.
    """
    sub_all = df[df["metric"] == metric].copy()
    if sub_all.empty:
        print(f"  WARNING: no rows found for metric '{metric}'; nothing to plot.")
        return

    configs = sorted(
        sub_all[["config_n", "config_p"]]
        .drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_p"]), axis=1)
    )
    labels = [config_label(*c) for c in configs]
    x = np.arange(len(configs))
    width, offsets = bar_offsets(len(VARIANTS))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), 5))

    sub_all = sub_all.set_index(["config_n", "config_p"])

    for i, variant in enumerate(VARIANTS):
        med_col = f"{variant}_median"
        iqr_col = f"{variant}_iqr"

        medians = []
        yerr = []
        for c in configs:
            if c in sub_all.index and med_col in sub_all.columns:
                row = sub_all.loc[c]
                med = float(row[med_col])
                iqr = float(row[iqr_col])
            else:
                med, iqr = np.nan, np.nan
            medians.append(med)
            # Only a single IQR value is available per point (not
            # separate Q1/Q3), so the error bar is drawn symmetrically
            # as median +/- IQR/2.
            yerr.append(np.nan if np.isnan(iqr) else iqr / 2.0)

        bar_x = x + offsets[i]

        ax.bar(
            bar_x, medians, width,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
            alpha=0.87,
        )
        ax.errorbar(
            bar_x, medians,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("Configuration")
    ax.set_ylabel(f"Median {metric}")
    ax.set_title(f"Median {metric} by configuration and variant\n(error bars: IQR/2)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="upper left")

    save_pdf(fig, out_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot median metric (default: Avg LBD) vs configuration, with IQR error bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv", type=Path, required=True,
        help="Path to the input CSV file (wide format with "
             "metric,config_n,config_p,config,V0_median,V0_iqr,... columns).",
    )
    p.add_argument(
        "--metric", type=str, default="Avg LBD",
        help="Value of the 'metric' column to filter on and plot.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("figures/avg_lbd_by_config.pdf"),
        help="Output path for the generated PDF figure.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.csv} ...")
    df = pd.read_csv(args.csv)

    print(f"Generating plot for metric '{args.metric}' ...")
    plot_metric_by_config(df, args.metric, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()