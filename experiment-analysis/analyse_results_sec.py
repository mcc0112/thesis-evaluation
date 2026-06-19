"""
analyse_results_primary.py  –  Four-variant circuit propagator analysis (primary benchmark)
============================================================================================

Loads one stats CSV per variant (V0–V3), computes derived/postprocessed statistics
per variant, then produces:

  1. Solve-status breakdown        – per (variant, n, k)
  2. PAR-2 scores                  – three-outcome, optimality-only
  3. Unified comparison tables     – all present variants together, one table per metric
  4. Internal / derived statistics – variant-specific, plotted only where applicable
  5. Cross-variant diagnostic plots (Hall set sizes, depth-at-failure, hall_circuit_fire_rate)
  6. Overall summary table         – one row per (variant, n, k) with key aggregates

Column schemas per variant
--------------------------
V0: instance, directory, status, num_solutions, wall_clock_time, cpu_time, failures,
    propagations, solveTime, nodes, peakDepth, restarts, nogoods, AverageLbd,
    AverageLearnedNogoodLength, AverageConflictSize, NumUnitNogoodsLearned,
    circuitNumberPropagationsThatFoundConflict, circuitNumberTotalFixedEdgesAtConflict,
    circuitNumberNumberOfConflicts, circuitNumberPropagationsThatPruned,
    circuitNumberPropagationsTotal, type, best_sol

V1: …V0 columns… + allDifferentNumberTotalHallSetSize, allDifferentNumberNumberOfConflicts,
    allDifferentNumberMaxHallSetSize, allDifferentNumberPropagationsTotal,
    allDifferentNumberPropagationsThatFoundConflict

V2: …V0 columns… + allDifferentNumberPropagationsTotal,
    allDifferentNumberPropagationsThatFoundConflict, allDifferentNumberPropagationsThatPruned,
    allDifferentNumberTotalValuesPruned, allDifferentNumberTotalHallSetSizeConflict,
    allDifferentNumberNumberOfConflicts, allDifferentNumberMaxHallSetSizeConflict,
    allDifferentNumberTotalPruningHallSetSize, allDifferentNumberNumberOfPruningExplanations,
    allDifferentNumberMaxPruningHallSetSize, allDifferentNumberTotalFixedEdgesAtConflict,
    allDifferentNumberTotalVarsInvolvedInPruning, allDifferentNumberMaxValuesPrunedFromSingleVar,
    allDifferentNumberTotalSccSizeAtPruning, allDifferentNumberNumberOfSccPruningCalls

V3: …V0 base columns… + circuitNumberPropagationsTotal, circuitNumberCircuitConflicts,
    circuitNumberAlldiffConflicts, circuitNumberTotalFixedEdgesAtConflict,
    circuitNumberNumberOfConflicts, circuitNumberGacPrunings,
    circuitNumberPropagationsThatGacPruned, circuitNumberTotalPruningHallSetSize,
    circuitNumberNumberOfPruningExplanations, circuitNumberMaxPruningHallSetSize,
    circuitNumberTotalSccSizeAtPruning, circuitNumberNumberOfSccPruningCalls,
    circuitNumberTotalVarsInvolvedInPruning, circuitNumberMaxValuesPrunedFromSingleVar,
    circuitNumberHallCircuitChecks, circuitNumberHallCircuitPrunings,
    circuitNumberCircuitPreventPrunings

PAR-2 scoring (three-outcome, minimisation)
-------------------------------------------
  optimal                          → actual wall_clock_time (or solveTime fallback)
  all other statuses               → 2 × timeout_limit

Aggregate statistics use MEDIAN + IQR throughout (heavy-tailed distributions).

Output structure
----------------
  results/
    analysis_<timestamp>/
      status_breakdown.csv
      par2_scores.csv
      summary_table.csv
      unified_<metric_group>.csv
      plots/
        status_breakdown.png
        par2.png
        runtime_*.png
        search_*.png
        expl_*.png
        internal_shared_*.png
        internal_v1_*.png
        internal_v2_*.png
        internal_v3_*.png
        cross_hall_set_sizes.png
        cross_depth_at_failure.png
        cross_hall_circuit_fire_rate.png

Usage
-----
    python experiment-analysis/analyse_results_primary.py [options]

Options
-------
  --v0  PATH        CSV for V0
  --v1  PATH        CSV for V1
  --v2  PATH        CSV for V2
  --v3  PATH        CSV for V3
  --out PATH        Output directory  (default: experiments/results/)
  --timeout INT     Timeout limit in seconds  (default: 1800)
  --benchmark STR   'primary' (default, 1800 s) or 'random' (300 s)
  --no-plots        Skip plot generation
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pandas and numpy are required:  pip install pandas numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    print("matplotlib not found – skipping plots")
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

VARIANTS = ["V0", "V1", "V2", "V3"]

VARIANT_LABELS = {"V0": "V0 (decomposed)", "V1": "V1 (conflict-only)",
                  "V2": "V2 (full GAC)",   "V3": "V3 (Hall-circuit)"}

VARIANT_COLORS = {"V0": "#4C72B0", "V1": "#DD8452",
                  "V2": "#C44E52", "V3": "#55A868"}

TIMEOUT_PRIMARY = 1800
TIMEOUT_RANDOM  = 300

MIN_SAMPLE_WARNING = 10

RUNTIME_COL = "solveTime"
WALL_COL    = "wall_clock_time"

OPTIMAL_STATUSES  = {"optimal"}
SOLUTION_STATUSES = {"optimal", "satisfiable", "timeout_with_solution"}
PENALTY_STATUSES  = {
    "satisfiable", "timeout_with_solution", "timeout_no_solution",
    "unsatisfiable", "unknown", "error",
}

# ---------------------------------------------------------------------------
# Expected columns per variant (for validation warnings)
# ---------------------------------------------------------------------------

V0_COLS = {
    "circuitNumberPropagationsThatFoundConflict",
    "circuitNumberTotalFixedEdgesAtConflict",
    "circuitNumberNumberOfConflicts",
    "circuitNumberPropagationsThatPruned",
    "circuitNumberPropagationsTotal",
}

V1_EXTRA_COLS = {
    "allDifferentNumberTotalHallSetSize",
    "allDifferentNumberNumberOfConflicts",
    "allDifferentNumberMaxHallSetSize",
    "allDifferentNumberPropagationsTotal",
    "allDifferentNumberPropagationsThatFoundConflict",
}

V2_EXTRA_COLS = {
    "allDifferentNumberPropagationsTotal",
    "allDifferentNumberPropagationsThatFoundConflict",
    "allDifferentNumberPropagationsThatPruned",
    "allDifferentNumberTotalValuesPruned",
    "allDifferentNumberTotalHallSetSizeConflict",
    "allDifferentNumberNumberOfConflicts",
    "allDifferentNumberMaxHallSetSizeConflict",
    "allDifferentNumberTotalPruningHallSetSize",
    "allDifferentNumberNumberOfPruningExplanations",
    "allDifferentNumberMaxPruningHallSetSize",
    "allDifferentNumberTotalFixedEdgesAtConflict",
    "allDifferentNumberTotalVarsInvolvedInPruning",
    "allDifferentNumberMaxValuesPrunedFromSingleVar",
    "allDifferentNumberTotalSccSizeAtPruning",
    "allDifferentNumberNumberOfSccPruningCalls",
}

V3_EXTRA_COLS = {
    "circuitNumberPropagationsTotal",
    "circuitNumberCircuitConflicts",
    "circuitNumberAlldiffConflicts",
    "circuitNumberTotalFixedEdgesAtConflict",
    "circuitNumberNumberOfConflicts",
    "circuitNumberGacPrunings",
    "circuitNumberPropagationsThatGacPruned",
    "circuitNumberTotalPruningHallSetSize",
    "circuitNumberNumberOfPruningExplanations",
    "circuitNumberMaxPruningHallSetSize",
    "circuitNumberTotalSccSizeAtPruning",
    "circuitNumberNumberOfSccPruningCalls",
    "circuitNumberTotalVarsInvolvedInPruning",
    "circuitNumberMaxValuesPrunedFromSingleVar",
    "circuitNumberHallCircuitChecks",
    "circuitNumberHallCircuitPrunings",
    "circuitNumberCircuitPreventPrunings",
}

VARIANT_REQUIRED_COLS = {
    "V0": V0_COLS,
    "V1": V0_COLS | V1_EXTRA_COLS,
    "V2": V0_COLS | V2_EXTRA_COLS,
    "V3": V3_EXTRA_COLS,
}

# ---------------------------------------------------------------------------
# Standard metric groups (column, ylabel, note, use_log_scale)
# ---------------------------------------------------------------------------

RUNTIME_METRICS = {
    "solve_time": ("solveTime",       "Solve time (s) [solver]",  "lower is better", False),
    "wall_time":  ("wall_clock_time", "Wall-clock time (s)",       "lower is better", False),
}

SEARCH_METRICS = {
    "failures":     ("failures",     "Failures (backtracks)", "lower is better", True),
    "nodes":        ("nodes",        "Nodes explored",        "lower is better", True),
    "propagations": ("propagations", "Propagations",          "lower is better", True),
    "peakDepth":    ("peakDepth",    "Peak search depth",     "informational",   False),
    "restarts":     ("restarts",     "Restarts",              "informational",   False),
}

EXPL_METRICS = {
    "lbd":           ("AverageLbd",                "Avg LBD",               "lower = stronger", False),
    "nogood_length": ("AverageLearnedNogoodLength", "Avg nogood length",     "lower = stronger", False),
    "nogoods":       ("nogoods",                   "Nogoods learned",       "informational",    True),
    "unit_nogoods":  ("NumUnitNogoodsLearned",      "Unit nogoods learned",  "higher = stronger", False),
    "conflict_size": ("AverageConflictSize",        "Avg conflict size",     "lower = cheaper",  False),
}

QUALITY_METRICS = {
    "best_objective": ("best_sol", "Best maxleg found", "lower is better", False),
}

# ---------------------------------------------------------------------------
# Internal / derived metric definitions
# (column name after postprocessing, ylabel, note, use_log, applicable variants)
# ---------------------------------------------------------------------------

INTERNAL_SHARED = {
    "avg_depth_at_failure": (
        "Avg depth at failure (fixed edges)", "lower = earlier detection", False,
        ["V0", "V1", "V2", "V3"],
    ),
    "circuit_conflict_rate": (
        "Circuit conflict detection rate", "informational", False,
        ["V0", "V1", "V2", "V3"],
    ),
    "circuit_pruning_rate": (
        "Circuit pruning rate", "informational", False,
        ["V0", "V1"],
    ),
}

INTERNAL_V1 = {
    "avg_hall_set_size": (
        "Avg Hall set size at conflict (V1)", "larger → higher LBD", False,
        ["V1"],
    ),
    "alldiff_conflict_detection_rate_v1": (
        "AllDiff conflict detection rate (V1)", "informational", False,
        ["V1"],
    ),
}

INTERNAL_V2 = {
    "avg_tight_hall_set_size": (
        "Avg tight Hall set size – pruning (V2)", "smaller → lower LBD", False,
        ["V2"],
    ),
    "avg_hall_set_size_conflict": (
        "Avg Hall set size at conflict (V2)", "informational", False,
        ["V2"],
    ),
    "avg_values_pruned_per_active_call": (
        "Avg values pruned per active call (V2)", "higher = more aggressive pruning", False,
        ["V2"],
    ),
    "avg_scc_size_when_pruning": (
        "Avg SCC size when pruning (V2)", "informational", False,
        ["V2"],
    ),
    "avg_vars_involved_per_pruning_call": (
        "Avg vars involved per pruning call (V2)", "close to 1 = concentrated", False,
        ["V2"],
    ),
    "alldiff_pruning_rate": (
        "AllDiff pruning rate (V2)", "informational", False,
        ["V2"],
    ),
    "alldiff_conflict_detection_rate_v2": (
        "AllDiff conflict detection rate (V2)", "informational", False,
        ["V2"],
    ),
}

INTERNAL_V3 = {
    "hall_circuit_fire_rate": (
        "Hall-circuit fire rate (V3)", "near-zero → V3 step rarely fires", False,
        ["V3"],
    ),
    "avg_tight_hall_set_size_v3": (
        "Avg tight Hall set size – pruning (V3)", "informational", False,
        ["V3"],
    ),
    "gac_pruning_rate": (
        "GAC pruning rate (V3)", "informational", False,
        ["V3"],
    ),
    "avg_scc_size_v3": (
        "Avg SCC size when pruning (V3)", "informational", False,
        ["V3"],
    ),
    "avg_vars_involved_v3": (
        "Avg vars involved per pruning call (V3)", "informational", False,
        ["V3"],
    ),
    "circuit_alldiff_conflict_rate": (
        "AllDiff conflict rate (V3)", "informational", False,
        ["V3"],
    ),
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where denominator is 0 or NaN."""
    b_safe = b.replace(0, np.nan)
    return a / b_safe


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def config_label(n, p) -> str:
    return f"n={int(n)}, p={int(p)}"


def optimal_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"].isin(OPTIMAL_STATUSES)]
    return df


def solution_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"].isin(SOLUTION_STATUSES)]
    return df


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = coerce_numerics(df)

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower().str.strip()
    if "best_sol" in df.columns:
        df["best_sol"] = pd.to_numeric(df["best_sol"], errors="coerce")

    # Parse n and k from instance name  e.g. "geo_n50_k10_seed3"
    if "instance" in df.columns:
        parsed = df["instance"].str.extract(r'n(\d+)_p(\d+)')
        df["config_n"] = pd.to_numeric(parsed[0], errors="coerce")
        df["config_p"] = pd.to_numeric(parsed[1], errors="coerce")

    df["propagator_variant"] = variant

    # Validate expected columns
    required = VARIANT_REQUIRED_COLS.get(variant, set())
    missing  = required - set(df.columns)
    if missing:
        print(f"  WARNING [{variant}]: missing expected columns: {sorted(missing)}")

    print(f"  [{variant}]  {len(df):4d} rows  ←  {path.name}")
    return df


def load_all(paths: dict) -> pd.DataFrame:
    frames = []
    for variant, path in paths.items():
        if path is not None:
            frames.append(load_csv(path, variant))
    if not frames:
        sys.exit("No CSV files loaded.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Postprocessing / derived statistics
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return column as Series or a NaN Series if absent."""
    if name in df.columns:
        return df[name].copy()
    return pd.Series(np.nan, index=df.index)


def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add postprocessed derived columns to df, computed per-row, guarded by variant.
    All raw divisions use safe_div (returns NaN on zero denominator).
    """
    df = df.copy()

    # ── V0 and V1 share the same circuit-side columns ──────────────────────
    mask_v01 = df["propagator_variant"].isin(["V0", "V1"])

    df.loc[mask_v01, "avg_depth_at_failure"] = safe_div(
        _col(df, "circuitNumberTotalFixedEdgesAtConflict"),
        _col(df, "circuitNumberNumberOfConflicts"),
    )[mask_v01]

    df.loc[mask_v01, "circuit_pruning_rate"] = safe_div(
        _col(df, "circuitNumberPropagationsThatPruned"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v01]

    df.loc[mask_v01, "circuit_conflict_rate"] = safe_div(
        _col(df, "circuitNumberPropagationsThatFoundConflict"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v01]

    # ── V1-specific AllDifferent columns ───────────────────────────────────
    mask_v1 = df["propagator_variant"] == "V1"

    df.loc[mask_v1, "avg_hall_set_size"] = safe_div(
        _col(df, "allDifferentNumberTotalHallSetSize"),
        _col(df, "allDifferentNumberNumberOfConflicts"),
    )[mask_v1]

    # pass-through (already a column, just give it a clean derived name)
    if "allDifferentNumberMaxHallSetSize" in df.columns:
        df.loc[mask_v1, "max_hall_set_size"] = df.loc[
            mask_v1, "allDifferentNumberMaxHallSetSize"]

    df.loc[mask_v1, "alldiff_conflict_detection_rate_v1"] = safe_div(
        _col(df, "allDifferentNumberPropagationsThatFoundConflict"),
        _col(df, "allDifferentNumberPropagationsTotal"),
    )[mask_v1]

    # ── V2-specific AllDifferent columns ───────────────────────────────────
    mask_v2 = df["propagator_variant"] == "V2"

    df.loc[mask_v2, "avg_depth_at_failure"] = safe_div(
        _col(df, "allDifferentNumberTotalFixedEdgesAtConflict"),
        _col(df, "allDifferentNumberNumberOfConflicts"),
    )[mask_v2]

    df.loc[mask_v2, "circuit_conflict_rate"] = safe_div(
        _col(df, "circuitNumberPropagationsThatFoundConflict"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v2]

    df.loc[mask_v2, "circuit_pruning_rate"] = safe_div(
        _col(df, "circuitNumberPropagationsThatPruned"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v2]

    df.loc[mask_v2, "alldiff_conflict_detection_rate_v2"] = safe_div(
        _col(df, "allDifferentNumberPropagationsThatFoundConflict"),
        _col(df, "allDifferentNumberPropagationsTotal"),
    )[mask_v2]

    df.loc[mask_v2, "alldiff_pruning_rate"] = safe_div(
        _col(df, "allDifferentNumberPropagationsThatPruned"),
        _col(df, "allDifferentNumberPropagationsTotal"),
    )[mask_v2]

    df.loc[mask_v2, "avg_values_pruned_per_active_call"] = safe_div(
        _col(df, "allDifferentNumberTotalValuesPruned"),
        _col(df, "allDifferentNumberPropagationsThatPruned"),
    )[mask_v2]

    df.loc[mask_v2, "avg_hall_set_size_conflict"] = safe_div(
        _col(df, "allDifferentNumberTotalHallSetSizeConflict"),
        _col(df, "allDifferentNumberNumberOfConflicts"),
    )[mask_v2]

    df.loc[mask_v2, "avg_tight_hall_set_size"] = safe_div(
        _col(df, "allDifferentNumberTotalPruningHallSetSize"),
        _col(df, "allDifferentNumberNumberOfPruningExplanations"),
    )[mask_v2]

    df.loc[mask_v2, "avg_scc_size_when_pruning"] = safe_div(
        _col(df, "allDifferentNumberTotalSccSizeAtPruning"),
        _col(df, "allDifferentNumberNumberOfSccPruningCalls"),
    )[mask_v2]

    df.loc[mask_v2, "avg_vars_involved_per_pruning_call"] = safe_div(
        _col(df, "allDifferentNumberTotalVarsInvolvedInPruning"),
        _col(df, "allDifferentNumberPropagationsThatPruned"),
    )[mask_v2]

    if "allDifferentNumberMaxValuesPrunedFromSingleVar" in df.columns:
        df.loc[mask_v2, "max_values_pruned_from_single_var"] = df.loc[
            mask_v2, "allDifferentNumberMaxValuesPrunedFromSingleVar"]

    # ── V3-specific columns (all under circuitNumber prefix) ───────────────
    mask_v3 = df["propagator_variant"] == "V3"

    df.loc[mask_v3, "circuit_conflict_rate"] = safe_div(
        _col(df, "circuitNumberCircuitConflicts"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v3]

    df.loc[mask_v3, "circuit_alldiff_conflict_rate"] = safe_div(
        _col(df, "circuitNumberAlldiffConflicts"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v3]

    df.loc[mask_v3, "avg_depth_at_failure"] = safe_div(
        _col(df, "circuitNumberTotalFixedEdgesAtConflict"),
        _col(df, "circuitNumberNumberOfConflicts"),
    )[mask_v3]

    df.loc[mask_v3, "gac_pruning_rate"] = safe_div(
        _col(df, "circuitNumberPropagationsThatGacPruned"),
        _col(df, "circuitNumberPropagationsTotal"),
    )[mask_v3]

    df.loc[mask_v3, "avg_values_pruned_per_call"] = safe_div(
        _col(df, "circuitNumberGacPrunings"),
        _col(df, "circuitNumberPropagationsThatGacPruned"),
    )[mask_v3]

    df.loc[mask_v3, "avg_tight_hall_set_size_v3"] = safe_div(
        _col(df, "circuitNumberTotalPruningHallSetSize"),
        _col(df, "circuitNumberNumberOfPruningExplanations"),
    )[mask_v3]

    df.loc[mask_v3, "avg_scc_size_v3"] = safe_div(
        _col(df, "circuitNumberTotalSccSizeAtPruning"),
        _col(df, "circuitNumberNumberOfSccPruningCalls"),
    )[mask_v3]

    df.loc[mask_v3, "avg_vars_involved_v3"] = safe_div(
        _col(df, "circuitNumberTotalVarsInvolvedInPruning"),
        _col(df, "circuitNumberPropagationsThatGacPruned"),
    )[mask_v3]

    df.loc[mask_v3, "hall_circuit_fire_rate"] = safe_div(
        _col(df, "circuitNumberHallCircuitPrunings"),
        _col(df, "circuitNumberHallCircuitChecks"),
    )[mask_v3]

    return df


# ---------------------------------------------------------------------------
# Status breakdown
# ---------------------------------------------------------------------------

ALL_STATUSES = [
    "optimal", "satisfiable", "timeout_with_solution",
    "timeout_no_solution", "unsatisfiable", "unknown", "error",
]


def compute_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (variant, n, p), grp in df.groupby(
            ["propagator_variant", "config_n", "config_p"]):
        total = len(grp)
        row = {
            "variant": variant, "config_n": n, "config_p": p,
            "config": config_label(n, p), "total": total,
        }
        for s in ALL_STATUSES:
            row[s] = int((grp["status"] == s).sum())
        row["proven_optimal_rate"] = (
            round(row["optimal"] / total, 4) if total > 0 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "config_n", "config_p"])


# ---------------------------------------------------------------------------
# PAR-2
# ---------------------------------------------------------------------------

def compute_par2(df: pd.DataFrame, timeout_limit: int) -> pd.DataFrame:
    penalty = 2 * timeout_limit

    if WALL_COL in df.columns and df[WALL_COL].notna().any():
        time_col = WALL_COL
    elif RUNTIME_COL in df.columns and df[RUNTIME_COL].notna().any():
        print(f"  WARNING: '{WALL_COL}' not found – using '{RUNTIME_COL}' for PAR-2.")
        time_col = RUNTIME_COL
    else:
        print("  WARNING: No time column found; PAR-2 cannot be computed.")
        return pd.DataFrame()

    def par2_val(row):
        status = row.get("status", "unknown")
        if status in OPTIMAL_STATUSES or status == "unsatisfiable":
            try:
                return float(row[time_col])
            except (TypeError, ValueError):
                return penalty
        return penalty

    df = df.copy()
    df["par2"] = df.apply(par2_val, axis=1)

    rows = []
    for (variant, n, p), grp in df.groupby(
            ["propagator_variant", "config_n", "config_p"]):
        vals = grp["par2"].dropna()
        n_opt = int((grp["status"] == "optimal").sum()) if "status" in grp else 0
        rows.append({
            "variant":       variant,
            "config_n":      n,
            "config_p":      p,
            "config":        config_label(n, p),
            "median_par2":   round(float(vals.median()), 4) if len(vals) else float("nan"),
            "n_instances":   len(grp),
            "n_optimal":     n_opt,
            "par2_time_col": time_col,
        })
    return pd.DataFrame(rows).sort_values(["variant", "config_n", "config_p"])


# ---------------------------------------------------------------------------
# Aggregate (median + IQR)
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, col: str,
              row_filter_fn=None) -> pd.DataFrame:
    if row_filter_fn is None:
        row_filter_fn = optimal_rows
    filtered = row_filter_fn(df)
    if col not in filtered.columns:
        return pd.DataFrame(columns=[
            "propagator_variant", "config_n", "config_p", "median", "iqr", "n"])
    rows = []
    for (variant, n, p), grp in filtered.groupby(
            ["propagator_variant", "config_n", "config_p"]):
        vals = grp[col].dropna()
        n_solved = len(vals)
        if n_solved == 0:
            continue
        total_in_cell = len(df[
            (df["propagator_variant"] == variant) &
            (df["config_n"] == n) & (df["config_p"] == p)])
        if n_solved < MIN_SAMPLE_WARNING:
            print(f"  WARNING: [{variant}] {config_label(n, p)} – only "
                  f"{n_solved}/{total_in_cell} rows pass filter for '{col}'. "
                  "Median may be unreliable.")
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        rows.append({
            "propagator_variant": variant,
            "config_n": n, "config_p": p,
            "config":   config_label(n, p),
            "median":   round(float(vals.median()), 4),
            "iqr":      round(q75 - q25, 4),
            "n":        n_solved,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unified comparison table (all variants, one metric)
# ---------------------------------------------------------------------------

def build_unified_table(df: pd.DataFrame, col: str,
                        variants: list, metric_label: str,
                        row_filter_fn=None) -> pd.DataFrame:
    sub  = df[df["propagator_variant"].isin(variants)]
    agg  = aggregate(sub, col, row_filter_fn=row_filter_fn)
    if agg.empty:
        return pd.DataFrame()
    all_configs = sorted(
        agg[["config_n", "config_p"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_p"]), axis=1))
    rows = []
    for (n, p) in all_configs:
        row = {"config_n": n, "config_p": p, "config": config_label(n, p)}
        for v in variants:
            s = agg[(agg["propagator_variant"] == v) &
                    (agg["config_n"] == n) & (agg["config_p"] == p)]
            if len(s):
                row[f"{v}_median"] = s.iloc[0]["median"]
                row[f"{v}_iqr"]    = s.iloc[0]["iqr"]
                row[f"{v}_n"]      = s.iloc[0]["n"]
            else:
                row[f"{v}_median"] = float("nan")
                row[f"{v}_iqr"]    = float("nan")
                row[f"{v}_n"]      = 0
        rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["metric"] = metric_label
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame, par2_df: pd.DataFrame,
                        status_df: pd.DataFrame,
                        timeout_limit: int) -> pd.DataFrame:
    """
    One row per (variant, config_n, config_k) with key aggregate statistics:
      - n_instances, n_optimal, proven_optimal_rate
      - median PAR-2
      - median failures, propagations, AverageLbd (optimal rows only)
      - median avg_depth_at_failure (optimal rows only, where available)
      - median avg_tight_hall_set_size / avg_hall_set_size (where available)
      - median hall_circuit_fire_rate (V3 only)
    """
    rows = []
    present_variants = sorted(df["propagator_variant"].unique())

    for (variant, n, p), grp in df.groupby(
            ["propagator_variant", "config_n", "config_p"]):
        opt_grp = grp[grp["status"].isin(OPTIMAL_STATUSES)] \
            if "status" in grp.columns else grp

        def med(col):
            if col in opt_grp.columns:
                v = opt_grp[col].dropna()
                return round(float(v.median()), 4) if len(v) else float("nan")
            return float("nan")

        # PAR-2 lookup
        par2_val = float("nan")
        if not par2_df.empty:
            pae = par2_df[(par2_df["variant"] == variant) &
                        (par2_df["config_n"] == n) &
                        (par2_df["config_p"] == p)]
            if len(pae):
                par2_val = float(pae.iloc[0]["median_par2"])

        # Status counts
        n_opt = int((grp["status"] == "optimal").sum()) if "status" in grp.columns else 0
        total = len(grp)

        row = {
            "variant":              variant,
            "config_n":             int(n),
            "config_p":             int(p),
            "config":               config_label(n, p),
            "n_instances":          total,
            "n_optimal":            n_opt,
            "proven_optimal_rate":  round(n_opt / total, 4) if total else float("nan"),
            "median_par2_s":        par2_val,
            "median_failures":      med("failures"),
            "median_propagations":  med("propagations"),
            "median_avg_lbd":       med("AverageLbd"),
            "median_nogood_length": med("AverageLearnedNogoodLength"),
            "median_wall_time_s":   med(WALL_COL),
            # Internal stats (NaN when not applicable for this variant)
            "median_avg_depth_at_failure":               med("avg_depth_at_failure"),
            "median_circuit_conflict_rate":              med("circuit_conflict_rate"),
            "median_avg_hall_set_size_v1":               med("avg_hall_set_size"),
            "median_avg_hall_set_size_conflict_v2":      med("avg_hall_set_size_conflict"),
            "median_avg_tight_hall_set_size_v2":         med("avg_tight_hall_set_size"),
            "median_avg_tight_hall_set_size_v3":         med("avg_tight_hall_set_size_v3"),
            "median_hall_circuit_fire_rate_v3":          med("hall_circuit_fire_rate"),
            "median_gac_pruning_rate_v3":                med("gac_pruning_rate"),
            "median_alldiff_pruning_rate_v2":            med("alldiff_pruning_rate"),
            "median_avg_vars_involved_per_pruning_v2":   med("avg_vars_involved_per_pruning_call"),
            "median_avg_scc_size_when_pruning_v2":       med("avg_scc_size_when_pruning"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["variant", "config_n", "config_p"])
    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"      Saved: {path.name}")
    plt.close(fig)


def plot_metric(df: pd.DataFrame, col: str, variants: list,
                ylabel: str, title: str, out_path: Path,
                use_log: bool = False,
                row_filter_fn=None) -> None:
    """Grouped bar chart with IQR error bars."""
    active_variants = [v for v in variants if v in df["propagator_variant"].unique()]
    agg = aggregate(
        df[df["propagator_variant"].isin(active_variants)],
        col, row_filter_fn=row_filter_fn)
    if agg.empty:
        print(f"      Skipping {out_path.name}: no data for '{col}'")
        return

    all_configs = sorted(
        agg[["config_n", "config_p"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_p"]), axis=1))
    config_labels = [config_label(*c) for c in all_configs]
    x = np.arange(len(all_configs))
    n_v   = len(active_variants)
    width = 0.8 / n_v
    offsets = (np.linspace(-(n_v - 1) / 2, (n_v - 1) / 2, n_v) * width)

    fig, ax = plt.subplots(figsize=(max(8, len(all_configs) * 1.2), 5))
    for i, v in enumerate(active_variants):
        sub = agg[agg["propagator_variant"] == v].set_index(["config_n", "config_p"])
        medians, iqrs = [], []
        for c in all_configs:
            if c in sub.index:
                medians.append(sub.loc[c, "median"])
                iqrs.append(sub.loc[c, "iqr"] / 2)
            else:
                medians.append(0); iqrs.append(0)
        ax.bar(x + offsets[i], medians, width,
               label=VARIANT_LABELS.get(v, v),
               color=VARIANT_COLORS.get(v, "#999999"), alpha=0.82)
        ax.errorbar(x + offsets[i], medians, yerr=iqrs,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8,
                                           rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    else:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    _save_fig(fig, out_path)


def plot_internal_stat(df: pd.DataFrame, col: str, applicable_variants: list,
                       ylabel: str, title: str, out_path: Path,
                       use_log: bool = False) -> None:
    """Same as plot_metric but restricted to applicable variants and uses optimal_rows."""
    present = [v for v in applicable_variants
               if v in df["propagator_variant"].unique()
               and col in df[df["propagator_variant"] == v].columns
               and df[df["propagator_variant"] == v][col].notna().any()]
    if not present:
        print(f"      Skipping {out_path.name}: '{col}' not present/non-NaN "
              f"for any of {applicable_variants}")
        return
    plot_metric(df, col, present, ylabel, title, out_path,
                use_log=use_log, row_filter_fn=optimal_rows)


def plot_par2(par2_df: pd.DataFrame, variants: list,
              timeout_limit: int, out_path: Path) -> None:
    if par2_df.empty:
        return
    sub = par2_df[par2_df["variant"].isin(variants)]
    if sub.empty:
        return
    all_configs = sorted(
        sub[["config_n", "config_p"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_p"]), axis=1))
    config_labels = [config_label(*c) for c in all_configs]
    x = np.arange(len(all_configs))
    n_v   = len(variants)
    width = 0.8 / n_v
    offsets = (np.linspace(-(n_v - 1) / 2, (n_v - 1) / 2, n_v) * width)
    penalty = 2 * timeout_limit

    fig, ax = plt.subplots(figsize=(max(8, len(all_configs) * 1.2), 5))
    for i, v in enumerate(variants):
        vsub = sub[sub["variant"] == v].set_index(["config_n", "config_p"])
        vals = [float(vsub.loc[c, "median_par2"]) if c in vsub.index else 0
                for c in all_configs]
        ax.bar(x + offsets[i], vals, width, label=VARIANT_LABELS.get(v, v),
               color=VARIANT_COLORS.get(v, "#999999"), alpha=0.82)

    ax.axhline(y=penalty, color="black", linestyle="--", linewidth=1,
               label=f"Timeout penalty ({penalty} s)")
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8,
                                           rotation=35, ha="right")
    ax.set_ylabel("Median PAR-2 (s)")
    ax.set_title(f"PAR-2 scores – all variants (penalty = {penalty} s)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    _save_fig(fig, out_path)


def plot_status_breakdown(status_df: pd.DataFrame, variants: list,
                          out_path: Path) -> None:
    if status_df.empty:
        return
    status_colors = {
        "optimal":               "#55A868",
        "satisfiable":           "#8DC87C",
        "timeout_with_solution": "#DD8452",
        "timeout_no_solution":   "#C44E52",
        "unsatisfiable":         "#9B59B6",
        "unknown":               "#95A5A6",
        "error":                 "#2C3E50",
    }
    sub = status_df[status_df["variant"].isin(variants)]
    if sub.empty:
        return
    all_configs = sorted(
        sub[["config_n", "config_p"]].drop_duplicates()
        .apply(lambda r: (r["config_n"], r["config_p"]), axis=1))
    config_labels = [config_label(*c) for c in all_configs]
    x = np.arange(len(all_configs))
    n_v   = len(variants)
    width = 0.8 / n_v
    offsets = (np.linspace(-(n_v - 1) / 2, (n_v - 1) / 2, n_v) * width)

    fig, ax = plt.subplots(figsize=(max(8, len(all_configs) * 1.4), 5))
    for i, v in enumerate(variants):
        vsub = sub[sub["variant"] == v].set_index(["config_n", "config_p"])
        bottoms = np.zeros(len(all_configs))
        for s in ALL_STATUSES:
            if s not in vsub.columns:
                continue
            heights = np.array([
                int(vsub.loc[c, s]) if c in vsub.index else 0
                for c in all_configs], dtype=float)
            ax.bar(x + offsets[i], heights, width, bottom=bottoms,
                   color=status_colors.get(s, "#cccccc"), alpha=0.85,
                   label=s if i == 0 else "_nolegend_")
            bottoms += heights
        ax.text(x[0] + offsets[i], -1.5, VARIANT_LABELS.get(v, v),
                ha="center", va="top", fontsize=7, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8,
                                           rotation=35, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("Solve status breakdown – all variants", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    _save_fig(fig, out_path)


def plot_cross_hall_set_sizes(df: pd.DataFrame, out_path: Path) -> None:
    """
    Cross-variant comparison: V1 avg_hall_set_size vs
    V2 avg_hall_set_size_conflict vs V2 avg_tight_hall_set_size vs
    V3 avg_tight_hall_set_size_v3.
    Each series plotted as a line over sorted configs so they overlay cleanly.
    """
    series_spec = [
        ("V1", "avg_hall_set_size",          "V1 – Hall set size (conflict)",  "#DD8452", "-"),
        ("V2", "avg_hall_set_size_conflict",  "V2 – Hall set size (conflict)",  "#C44E52", "--"),
        ("V2", "avg_tight_hall_set_size",     "V2 – tight Hall set (pruning)",  "#4C72B0", "-"),
        ("V3", "avg_tight_hall_set_size_v3",  "V3 – tight Hall set (pruning)",  "#55A868", "--"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    any_plotted = False
    all_configs_set: set = set()

    # Collect all (n,k) configs across relevant variants
    for (variant, col, label, color, ls) in series_spec:
        mask = (df["propagator_variant"] == variant) & df[col].notna() \
            if col in df.columns else pd.Series(False, index=df.index)
        if not mask.any():
            continue
        sub = df[mask].groupby(["config_n", "config_p"])[col].median()
        for idx in sub.index:
            all_configs_set.add(idx)

    all_configs = sorted(all_configs_set)
    if not all_configs:
        print(f"      Skipping {out_path.name}: no Hall-set-size data found")
        plt.close(fig)
        return

    x = np.arange(len(all_configs))
    config_labels = [config_label(*c) for c in all_configs]

    for (variant, col, label, color, ls) in series_spec:
        if col not in df.columns:
            continue
        mask = df["propagator_variant"] == variant
        sub  = df[mask].groupby(["config_n", "config_p"])[col].median()
        ys   = [float(sub.loc[c]) if c in sub.index else float("nan")
                for c in all_configs]
        if any(not np.isnan(y) for y in ys):
            ax.plot(x, ys, marker="o", linestyle=ls, color=color,
                    label=label, linewidth=1.8, markersize=5)
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8,
                                           rotation=35, ha="right")
    ax.set_ylabel("Median Hall set size")
    ax.set_title("Cross-variant Hall set sizes\n"
                 "(lower = cheaper explanations, lower expected LBD)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.45)
    _save_fig(fig, out_path)


def plot_cross_depth_at_failure(df: pd.DataFrame, out_path: Path) -> None:
    """V0 / V1 / V2 / V3 avg_depth_at_failure on one chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    all_configs_set: set = set()
    for v in ["V0", "V1", "V2", "V3"]:
        if "avg_depth_at_failure" not in df.columns:
            continue
        mask = (df["propagator_variant"] == v) & df["avg_depth_at_failure"].notna()
        for idx in df[mask].groupby(["config_n", "config_p"]).groups:
            all_configs_set.add(idx)

    all_configs = sorted(all_configs_set)
    if not all_configs:
        print(f"      Skipping {out_path.name}: no avg_depth_at_failure data")
        plt.close(fig); return

    x = np.arange(len(all_configs))
    any_plotted = False
    for v in ["V0", "V1", "V2", "V3"]:
        if v not in df["propagator_variant"].unique():
            continue
        if "avg_depth_at_failure" not in df.columns:
            continue
        mask = (df["propagator_variant"] == v)
        sub  = df[mask & df["avg_depth_at_failure"].notna()].groupby(
            ["config_n", "config_p"])["avg_depth_at_failure"].median()
        ys = [float(sub.loc[c]) if c in sub.index else float("nan")
              for c in all_configs]
        if any(not np.isnan(y) for y in ys):
            ax.plot(x, ys, marker="o", linestyle="-",
                    color=VARIANT_COLORS[v],
                    label=VARIANT_LABELS[v], linewidth=1.8, markersize=5)
            any_plotted = True

    if not any_plotted:
        plt.close(fig); return

    ax.set_xticks(x)
    ax.set_xticklabels([config_label(*c) for c in all_configs],
                       fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Median avg depth at failure (fixed edges)")
    ax.set_title("Cross-variant: avg depth at failure\n"
                 "(lower = earlier conflict detection)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.45)
    _save_fig(fig, out_path)


def plot_hall_circuit_fire_rate(df: pd.DataFrame, out_path: Path) -> None:
    """V3-only: hall_circuit_fire_rate across configs."""
    plot_internal_stat(
        df, "hall_circuit_fire_rate", ["V3"],
        ylabel="Hall-circuit fire rate",
        title="V3 Hall-circuit fire rate\n"
              "(near zero → V3 extra step rarely fires → explains underperformance)",
        out_path=out_path,
        use_log=False,
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, out_dir: Path, timeout_limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    present_variants = [v for v in VARIANTS
                        if v in df["propagator_variant"].unique()]
    print(f"\nPresent variants: {present_variants}")

    # ── 0. Derived / postprocessed statistics ─────────────────────────────
    print("\n=== Computing derived statistics ===")
    df = compute_derived_stats(df)

    # ── 1. Status breakdown ───────────────────────────────────────────────
    print("\n=== Solve status breakdown ===")
    status_df = compute_status_breakdown(df)
    if not status_df.empty:
        print(status_df.to_string(index=False))
        status_df.to_csv(out_dir / "status_breakdown.csv", index=False)
        if HAS_MPL:
            plot_status_breakdown(
                status_df, present_variants,
                plot_dir / "status_breakdown.png")

    # ── 2. PAR-2 scores ───────────────────────────────────────────────────
    print("\n=== PAR-2 scores (optimality-only) ===")
    par2_df = compute_par2(df, timeout_limit)
    if not par2_df.empty:
        print(par2_df.to_string(index=False))
        par2_df.to_csv(out_dir / "par2_scores.csv", index=False)
        if HAS_MPL:
            plot_par2(par2_df, present_variants, timeout_limit,
                      plot_dir / "par2.png")

    # ── 3. Unified metric comparisons ────────────────────────────────────
    all_metric_groups = [
        ("runtime", RUNTIME_METRICS, optimal_rows),
        ("search",  SEARCH_METRICS,  optimal_rows),
        ("expl",    EXPL_METRICS,    optimal_rows),
        ("quality", QUALITY_METRICS, solution_rows),
    ]

    for group_name, metrics, row_filter_fn in all_metric_groups:
        print(f"\n{'='*65}")
        print(f"  Metric group: {group_name}  (all variants: {present_variants})")
        print(f"{'='*65}")
        combined_rows = []
        for metric_key, (col, ylabel, note, use_log) in metrics.items():
            tbl = build_unified_table(
                df, col, present_variants,
                metric_label=f"{ylabel} ({note})",
                row_filter_fn=row_filter_fn)
            if not tbl.empty:
                print(f"\n  ── {ylabel} ({note}) ──")
                display_cols = (
                    ["config"]
                    + [f"{v}_median" for v in present_variants if f"{v}_median" in tbl]
                    + [f"{v}_iqr"    for v in present_variants if f"{v}_iqr"    in tbl])
                print(tbl[display_cols].to_string(index=False))
                tbl.insert(0, "metric", ylabel)
                combined_rows.append(tbl)

            if HAS_MPL:
                plot_metric(
                    df, col, present_variants,
                    ylabel=f"Median {ylabel}",
                    title=f"{ylabel} ({note}) – all variants",
                    out_path=plot_dir / f"{group_name}_{metric_key}.png",
                    use_log=use_log,
                    row_filter_fn=row_filter_fn)

        if combined_rows:
            out_name = f"unified_{group_name}.csv"
            pd.concat(combined_rows, ignore_index=True).to_csv(
                out_dir / out_name, index=False)
            print(f"\n  Saved: {out_name}")

    # ── 4. Internal / variant-specific statistics ─────────────────────────
    print(f"\n{'='*65}")
    print("  Internal / derived statistics")
    print(f"{'='*65}")

    all_internal_groups = [
        ("internal_shared", INTERNAL_SHARED),
        ("internal_v1",     INTERNAL_V1),
        ("internal_v2",     INTERNAL_V2),
        ("internal_v3",     INTERNAL_V3),
    ]

    for group_name, metric_dict in all_internal_groups:
        for metric_key, spec in metric_dict.items():
            ylabel, note, use_log, applicable = spec
            col = metric_key  # derived column name matches dict key
            if HAS_MPL:
                plot_internal_stat(
                    df, col, applicable,
                    ylabel=f"Median {ylabel}",
                    title=f"{ylabel} ({note})",
                    out_path=plot_dir / f"{group_name}_{metric_key}.png",
                    use_log=use_log)

        # Save per-group CSV with all applicable derived cols
        relevant_cols = list(metric_dict.keys())
        all_applicable = set()
        for spec in metric_dict.values():
            all_applicable |= set(spec[3])
        sub = df[df["propagator_variant"].isin(all_applicable)].copy()
        existing = [c for c in relevant_cols if c in sub.columns]
        if existing:
            sub[["propagator_variant", "config_n", "config_p", "instance"]
                + existing].to_csv(
                out_dir / f"{group_name}_raw.csv", index=False)

    # ── 5. Cross-variant diagnostic plots ─────────────────────────────────
    if HAS_MPL:
        print("\n=== Cross-variant diagnostic plots ===")
        plot_cross_hall_set_sizes(df, plot_dir / "cross_hall_set_sizes.png")
        plot_cross_depth_at_failure(df, plot_dir / "cross_depth_at_failure.png")
        plot_hall_circuit_fire_rate(df, plot_dir / "cross_hall_circuit_fire_rate.png")

    # ── 6. Overall summary table ──────────────────────────────────────────
    print("\n=== Overall summary table ===")
    summary = build_summary_table(df, par2_df, status_df, timeout_limit)
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"\n  Saved: summary_table.csv")

    print("\nAnalysis complete.")
    print(f"Output directory: {out_dir}")


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_latest_csvs(results_dir: Path) -> dict:
    found = {}
    for v in VARIANTS:
        matches = sorted(results_dir.glob(f"stats_{v}_*.csv"))
        if len(matches) > 1:
            print(f"  WARNING: {len(matches)} CSVs found for {v}; "
                  f"using most recent: {matches[-1].name}")
        found[v] = matches[-1] if matches else None
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse circuit propagator primary benchmark results (V0–V3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--v0",        type=Path, default=None, help="CSV for V0")
    p.add_argument("--v1",        type=Path, default=None, help="CSV for V1")
    p.add_argument("--v2",        type=Path, default=None, help="CSV for V2")
    p.add_argument("--v3",        type=Path, default=None, help="CSV for V3")
    p.add_argument("--out",       type=Path, default=RESULTS_DIR,
                   help="Root output directory")
    p.add_argument("--timeout",   type=int,  default=None,
                   help="Timeout limit in seconds (overrides --benchmark default)")
    p.add_argument("--benchmark", type=str,  default="primary",
                   choices=["primary", "random"],
                   help="'primary' → 1800 s timeout; 'random' → 300 s timeout")
    p.add_argument("--no-plots",  action="store_true", help="Skip plot generation")
    return p.parse_args()


def main():
    args = parse_args()

    global HAS_MPL
    if args.no_plots:
        HAS_MPL = False

    # Resolve timeout
    if args.timeout is not None:
        timeout_limit = args.timeout
    else:
        timeout_limit = TIMEOUT_PRIMARY if args.benchmark == "primary" \
            else TIMEOUT_RANDOM
    print(f"Benchmark mode: {args.benchmark}  |  Timeout: {timeout_limit} s")

    explicit = {"V0": args.v0, "V1": args.v1, "V2": args.v2, "V3": args.v3}
    if any(v is not None for v in explicit.values()):
        paths = explicit
    else:
        print("No explicit CSVs provided – auto-discovering latest per variant …")
        paths = discover_latest_csvs(RESULTS_DIR)

    print("\nLoading CSVs:")
    available = {v: p for v, p in paths.items() if p is not None}
    if not available:
        sys.exit(f"No stats_V*_*.csv files found in {RESULTS_DIR}. "
                 "Run experiments first or provide --v0 / --v1 / --v2 / --v3.")

    for v, p in paths.items():
        if p is None:
            print(f"  [{v}]  not provided / not found – skipped")

    df = load_all(available)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"analysis_{ts}"

    print(f"\n{'='*65}")
    print(f"  Total rows loaded:  {len(df)}")
    print(f"  Variants present:   {sorted(df['propagator_variant'].unique())}")
    print(f"  Output:             {out_dir}")
    print(f"{'='*65}")

    run_analysis(df, out_dir, timeout_limit)


if __name__ == "__main__":
    main()