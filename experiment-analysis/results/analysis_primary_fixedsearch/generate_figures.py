#!/usr/bin/env python3
"""
generate_figures.py
-------------------
Produces the four paper figures for Experiment 1 (primary benchmark)
as individual PDFs.

  F1_status_breakdown.pdf   – stacked bar, all variants, all configs
  F2_par2_n50.pdf           – PAR-2 grouped bar, n=50 configs only
  F3_search_metrics_n20.pdf – two-panel propagations + failures, n=20 only
  F4_hall_circuit_fire_rate.pdf – V3 fire rate line chart

Usage:
    python generate_figures.py \
        --summary summary_table.csv \
        --status  status_breakdown.csv \
        --par2    par2_scores.csv \
        --out     figures/

All three CSVs are produced by analyse_results_primary.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------
# Shared style
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


def config_label(n, k):
    return f"n={int(n)}, k={int(k)}"


def save_pdf(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def bar_offsets(n_variants):
    width = 0.8 / n_variants
    offsets = (
        np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants)
        * width
    )
    return width, offsets


# ------------------------------------------------------------------
# F1 – Status breakdown (all configs, all variants)
# ------------------------------------------------------------------

def plot_f1_status_breakdown(status_df: pd.DataFrame, out_path: Path):
    """
    Stacked bar chart of solve status per (variant, config).
    Variant labels removed from x-axis; grouping explained in caption.
    """
    df = status_df.copy()

    df["timeout_no_solution"] = (
        df["timeout_no_solution"]
        + df.get("error",   pd.Series(0, index=df.index))
        + df.get("unknown", pd.Series(0, index=df.index))
    )

    status_stack = ["optimal", "satisfiable", "timeout_no_solution"]
    status_colors = {
        "optimal":              "#55A868",
        "satisfiable":          "#8DC87C",
        "timeout_no_solution":  "#C44E52",
    }
    status_nicenames = {
        "optimal":             "Optimal",
        "satisfiable":         "Satisfiable (timeout)",
        "timeout_no_solution": "Timeout / failed",
    }

    configs = sorted(
        df[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )
    x = np.arange(len(configs))
    width, offsets = bar_offsets(len(VARIANTS))

    fig, ax = plt.subplots(figsize=(max(14, len(configs) * 0.85), 5))

    for i, variant in enumerate(VARIANTS):
        sub = (
            df[df["variant"] == variant]
            .set_index(["config_n", "config_k"])
        )
        bottoms = np.zeros(len(configs))
        for status in status_stack:
            heights = np.array([
                float(sub.loc[c, status]) if c in sub.index else 0.0
                for c in configs
            ])
            ax.bar(
                x + offsets[i], heights, width,
                bottom=bottoms,
                color=status_colors[status],
                alpha=0.9,
                label=status_nicenames[status] if i == 0 else "_nolegend_",
            )
            bottoms += heights

    # Group labels: one label per config group, centred under the 4 bars
    ax.set_xticks(x)
    ax.set_xticklabels(
        [config_label(*c) for c in configs],
        rotation=40, ha="right"
    )

    # Add variant order indicator once, below the first group
    # as a compact text annotation
    variant_order_str = " | ".join(
        [f"({j+1}) {VARIANT_LABELS[v]}" for j, v in enumerate(VARIANTS)]
    )
    ax.annotate(
        f"Bar order within each group: {variant_order_str}",
        xy=(0.01, -0.22), xycoords="axes fraction",
        fontsize=7, color="dimgray",
    )

    ax.set_ylabel("Number of instances")
    ax.set_xlabel("Configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="upper right")
    ax.set_title("Solve status breakdown by configuration and variant")

    save_pdf(fig, out_path)


# ------------------------------------------------------------------
# F2 – PAR-2 scores, n=50 only
# ------------------------------------------------------------------

def plot_f2_par2_n50(par2_df: pd.DataFrame, timeout_limit: int,
                     out_path: Path):
    df = par2_df[par2_df["config_n"] == 50].copy()
    if df.empty:
        print("  WARNING: no n=50 rows in PAR-2 data; skipping F2.")
        return

    configs = sorted(
        df[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )
    labels = [f"k={int(k)}" for _, k in configs]
    x = np.arange(len(configs))
    width, offsets = bar_offsets(len(VARIANTS))
    penalty = 2 * timeout_limit

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, variant in enumerate(VARIANTS):
        sub = (
            df[df["variant"] == variant]
            .set_index(["config_n", "config_k"])
        )
        vals = [
            float(sub.loc[c, "median_par2"]) if c in sub.index else np.nan
            for c in configs
        ]
        ax.bar(
            x + offsets[i], vals, width,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
            alpha=0.87,
        )

    ax.axhline(
        y=penalty, color="black", linestyle="--", linewidth=1.2,
        label=f"Penalty ceiling ({penalty:,} s)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k  (n = 50)")
    ax.set_ylabel("Median PAR-2 (s)")
    ax.set_title("PAR-2 scores at n = 50 (lower is better)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    save_pdf(fig, out_path)


# ------------------------------------------------------------------
# F3 – Propagations + Failures, n=20 only, log scale, with IQR error bars
# ------------------------------------------------------------------

def plot_f3_search_n20(summary_df: pd.DataFrame, error_df: pd.DataFrame, out_path: Path):

    """
    Two-panel grouped bar chart (log scale) with IQR error bars.
    Left:  median propagations   Right: median failures
    Restricted to n=20 (all variants solve all instances -> clean comparison).

    Expects columns in summary_df:
      median_propagations, q1_propagations, q3_propagations
      median_failures,     q1_failures,     q3_failures
    """
    df = summary_df[summary_df["config_n"] == 20].copy()
    if df.empty:
        print("  WARNING: no n=20 rows in summary data; skipping F3.")
        return

    err_df = error_df[error_df["config_n"] == 20].copy()
    if err_df.empty:
        print("  WARNING: no n=20 rows in error-bar data; bars will be drawn without error bars.")

    configs = sorted(
        df[["config_n", "config_k"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_k"]), axis=1)
    )
    labels = [f"k={int(k)}" for _, k in configs]
    x = np.arange(len(configs))
    width, offsets = bar_offsets(len(VARIANTS))

    metrics = [
        (
            "median_propagations",
            "Propagations",            # metric name in error_df
            "Propagations (log scale)",
            "Median propagations",
        ),
        (
            "median_failures",
            "Failures (backtracks)",   # metric name in error_df
            "Failures (log scale)",
            "Median failures",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (col_med, error_metric, ylabel, title) in zip(axes, metrics):
            # Rows in error_df relevant to this panel's metric
            err_sub_metric = err_df[err_df["metric"] == error_metric].set_index(
                ["config_n", "config_k"]
            )

            for i, variant in enumerate(VARIANTS):
                sub = (
                    df[df["variant"] == variant]
                    .set_index(["config_n", "config_k"])
                )
                iqr_col = f"{variant}_iqr"

                medians = []
                yerr = []

                for c in configs:
                    med = float(sub.loc[c, col_med]) if c in sub.index else np.nan
                    medians.append(med)

                    if c in err_sub_metric.index and iqr_col in err_sub_metric.columns:
                        iqr = float(err_sub_metric.loc[c, iqr_col])
                    else:
                        iqr = np.nan
                    # Only a single IQR value is available (not separate
                    # Q1/Q3), so the error bar is drawn symmetrically as
                    # median +/- IQR/2.
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

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("k  (n = 20)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    axes[1].legend(loc="upper left")

    save_pdf(fig, out_path)


# ------------------------------------------------------------------
# F4 – V3 Hall-circuit fire rate line chart
# ------------------------------------------------------------------

def plot_f4_fire_rate(summary_df: pd.DataFrame, out_path: Path):
    col = "median_hall_circuit_fire_rate_v3"
    df = (
        summary_df[
            (summary_df["variant"] == "V3") &
            summary_df[col].notna()
        ]
        .sort_values(["config_n", "config_k"])
        .copy()
    )

    if df.empty:
        print("  WARNING: no V3 fire-rate data; skipping F4.")
        return

    labels = [config_label(n, k) for n, k in zip(df["config_n"], df["config_k"])]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        x, df[col].values,
        marker="o", linewidth=2,
        color=VARIANT_COLORS["V3"],
        label="V3 (Hall-circuit)",
    )

    ax.axhline(
        y=0.10, color="grey", linestyle="--", linewidth=1,
        label="10% reference",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Median Hall-circuit fire rate")
    ax.set_xlabel("Configuration")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "V3 Hall-circuit pruning fire rate\n"
        "(fraction of checks where the entry-exit condition is satisfied)"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    save_pdf(fig, out_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate paper figures for Experiment 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--summary", type=Path, default=Path("summary_table.csv"))
    p.add_argument("--status",  type=Path, default=Path("status_breakdown.csv"))
    p.add_argument("--par2",    type=Path, default=Path("par2_scores.csv"))
    p.add_argument("--errors",  type=Path, default=Path("search_metrics_errors.csv"),
                   help="Wide-format CSV with per-variant IQR columns for "
                        "Propagations / Failures (backtracks), used for F3 error bars.")
    p.add_argument("--out",     type=Path, default=Path("figures"))
    p.add_argument("--timeout", type=int,  default=1800,
                   help="Timeout used in experiments (for PAR-2 penalty line)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    summary_df = pd.read_csv(args.summary)
    status_df  = pd.read_csv(args.status)
    par2_df    = pd.read_csv(args.par2)
    error_df   = pd.read_csv(args.errors)

    print("\nGenerating figures …")

    plot_f1_status_breakdown(
        status_df,
        args.out / "F1_status_breakdown.pdf",
    )

    plot_f2_par2_n50(
        par2_df,
        timeout_limit=args.timeout,
        out_path=args.out / "F2_par2_n50.pdf",
    )

    plot_f3_search_n20(
        summary_df,
        error_df,
        args.out / "F3_search_metrics_n20.pdf",
    )

    plot_f4_fire_rate(
        summary_df,
        args.out / "F4_hall_circuit_fire_rate.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()