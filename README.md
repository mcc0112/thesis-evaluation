# AllDifferent Inside Circuit — Evaluation Repository

Reproducibility package accompanying the Bachelor thesis:

> **AllDifferent Inside Circuit: An Experimental Evaluation of Propagation Strength
> and Explanation Quality in Lazy Clause Generation**
> Mireia Carrió Cortada
> Supervisors: Emir Demirović, Imko Marijnissen
> EEMCS, Delft University of Technology — CSE3000 Research Project, June 2026
>
> *An electronic version of the thesis will be available at
> [repository.tudelft.nl](http://repository.tudelft.nl/) once validation is complete.*

This repository contains everything needed to reproduce the experiments, figures,
and tables in the thesis: the four propagator variants (V0–V3), benchmark
generators, the benchmark instances themselves, SLURM experiment scripts used on
DelftBlue, and the analysis pipeline that turns raw solver output into the
tables/figures reported in the paper.

For the live development history of the solver itself (not squashed), see the
canonical development fork: https://github.com/mcc0112/Pumpkin

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Solver Variants](#solver-variants)
- [Setup](#setup)
- [Reproducing the Pipeline](#reproducing-the-pipeline)
  1. [Generate Benchmarks](#1-generate-benchmarks)
  2. [Build the Solver Variants](#2-build-the-solver-variants)
  3. [Flatten Instances & Build the Instance Database](#3-flatten-instances--build-the-instance-database)
  4. [Generate & Submit Experiments](#4-generate--submit-experiments)
  5. [Parse Logs](#5-parse-logs)
  6. [Aggregate Statistics](#6-aggregate-statistics)
  7. [Generate Tables & Figures](#7-generate-tables--figures)
- [Experiment Naming Scheme](#experiment-naming-scheme)
- [Experiment ↔ Thesis Section Mapping](#experiment--thesis-section-mapping)
- [Provenance](#provenance)

---

## Repository Structure

```
thesis-evaluation/
├── solvers/                        # The 4 propagator variants
│   ├── pumpkin-v0/                 # Decomposed AllDifferent
│   ├── pumpkin-v1/                 # Conflict-only matching
│   ├── pumpkin-v2/                 # Full GAC
│   └── pumpkin-v3/                 # Hall-circuit pruning
│
├── benchmark-generators/           # Instance generators
│   ├── generate_structured_instances.py   # Francis & Stuckey k-nearest geographic generator
│   └── generate_random_instances.py       # Erdős–Rényi random instance generator
│
├── benchmarks/                     # Generated .mzn instances (structured + random)
│   ├── primary/                    # Geographic k-nearest, n{20,50,100,150} x k{...}
│   └── secondary/                  # Erdős–Rényi random, n{20,50,100,150} x p{0.05,0.10,0.15}
│
├── flattened/                      # .fzn instances per solver variant + instances.csv metadata
│   ├── pumpkinv0/
│   ├── pumpkinv1/
│   ├── pumpkinv2/
│   ├── pumpkinv3/
│   └── instances.csv               # Per-instance metadata (vars, constraints, objective, type)
│
├── processed/                      # Optional: flattened instances rewritten to log objectives
│
├── experiments/                    # One subfolder per (benchmark x search x variant) run
│   ├── pri_fixed_pumpkin-v0/  ... pri_fixed_pumpkin-v3/   # Exp 1.1 — geographic, fixed search
│   ├── pri_vsids_pumpkin-v0/  ... pri_vsids_pumpkin-v3/   # Exp 1.2 — geographic, VSIDS search
│   └── sec_fixed_pumpkin-v0/  ... sec_fixed_pumpkin-v3/   # Exp 2   — random graphs, fixed search
│       ├── experiment.toml         # Auto-recorded git SHA + branch for this run (see Provenance)
│       ├── commands.txt            # One MiniZinc invocation per instance
│       ├── array_*.job             # SLURM job array script(s)
│       ├── submit_jobs.sh          # Submits all array jobs for this experiment
│       ├── slurm_output/           # SLURM stdout/stderr per array job
│       └── <run_id>/               # One folder per instance run: driver.log, output.log, output.err, stats.toml
│
├── raw_statistics/                 # Per-experiment-family aggregated stats, ready for analysis
│   ├── pri_fixed/
│   │   ├── statistics-v0.csv
│   │   ├── statistics-v1.csv
│   │   ├── statistics-v2.csv
│   │   └── statistics-v3.csv
│   ├── pri_vsids/
│   │   └── statistics-v{0,1,2,3}.csv
│   └── sec_fixed/
│       └── statistics-v{0,1,2,3}.csv
│
├── experiment-analysis/
│   ├── analyse_results_primary.py  # Produces tables/figures for Exp 1.1 + 1.2 (geographic)
│   ├── analyse_results_sec.py      # Produces tables/figures for Exp 2 (random)
│   └── results/                    # Output: per-experiment tables (median, IQR, PAR-2, internal stats)
│       └── <experiment>/
│           └── report_figures/     # Figures used directly in the thesis report
│
├── build-instance-database.py      # Validates + indexes flattened .fzn instances -> instances.csv
├── flatten.py                      # Flattens .mzn -> .fzn for one solver variant
├── generate-experiment.py          # Generates SLURM jobs for an experiment from instances.csv
├── parse-logs-no-internal-stats.py # Parses SLURM run logs -> stats.toml (external stats only)
├── parse-logs-internal-stats.py    # Same, plus variant-specific internal solver stats (see step 5)
├── aggregate-statistics.py         # Aggregates stats.toml files -> statistics_summary.csv per experiment
├── common.py                       # Shared config: SOLVER_DIRS, INSTANCES, CONFIGURATIONS, etc.
│
├── SOLVER_VERSIONS.md              # Pins each variant to an exact commit SHA
├── pyproject.toml
├── uv.lock
└── README.md                       # This file
```

> **Note on provenance:** the four variants under `solvers/` were imported via
> `git subtree` from the live development fork at
> `https://github.com/mcc0112/Pumpkin`.
> See [`SOLVER_VERSIONS.md`](./SOLVER_VERSIONS.md) and the
> [Provenance](#provenance) section below for how exact commits are tracked.

---

## Solver Variants

All four variants are implemented as propagators for `AllDifferent` embedded
within `Circuit`, inside the [Pumpkin](https://github.com/mcc0112/Pumpkin) LCG
solver. They form a strictly increasing chain of propagation strength, each
adding exactly one enhancement over its predecessor:

| Folder | Thesis name | Description |
|---|---|---|
| `solvers/pumpkin-v0/` | **V0** — Decomposed baseline | AllDifferent decomposed into pairwise inequalities; check-prevent subtour prevention (Francis & Stuckey). Weakest, cheapest per call. |
| `solvers/pumpkin-v1/` | **V1** — Conflict-only matching | Adds global Hall-set detection via Hopcroft–Karp maximum matching; reports infeasibility but does not prune individual values. |
| `solvers/pumpkin-v2/` | **V2** — Full GAC | Adds complete SCC-based domain pruning (Régin's algorithm via Tarjan's SCC) — removes values that cannot participate in any perfect matching, *before* branching. |
| `solvers/pumpkin-v3/` | **V3** — Hall-circuit pruning | Extends V2 with an additional pruning step exploiting circuit structure on tight Hall sets with a unique entry/exit (Bertagnon & Gavanelli, 2024). |

Each folder is a full, independently-buildable copy of the solver — see each
folder's own `README.md` for build instructions, since each is a snapshot of
a different branch of the underlying Rust codebase.

---

## Setup

This repo uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/ for other platforms)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies (resolves uv.lock)
uv sync
```

Requires Python ≥3.10 (see `pyproject.toml`). Core dependencies: `matplotlib`
(figures), `polars` (results processing), `tomli` / `tomli-w` (reading/writing
TOML experiment metadata and `stats.toml` run output), `tqdm` (progress bars
during aggregation).

You will also need:
- **MiniZinc** version 2.9.5 to compile `.mzn` → `.fzn` and to dispatch runs.
  Each solver variant is registered as its own MiniZinc solver configuration
  (`.msc` file) with a unique solver id — see [step 2](#2-build-the-solver-variants)
  for how this is set up per variant.
- **Rust / Cargo** to build each solver variant
- Access to a SLURM cluster to reproduce the exact DelftBlue runs (see
  [Cluster Details](#cluster-details)), or you can adapt
  `generate-experiment.py` to emit commands you run locally instead of via SLURM

---

## Reproducing the Pipeline

The full pipeline: **generate benchmarks → build solvers → flatten & index
instances → generate & submit SLURM experiments → parse logs → aggregate
statistics → generate tables/figures.**

### 1. Generate Benchmarks

Per configuration run (example for n=20, k=3 structured; n=20, p=0.05 random):

```bash
uv run python benchmark-generators/generate_structured_instances.py \
    -n 20 \
    -k 3 \
    -s 42 \
    -o ../benchmarks/primary

uv run python benchmark-generators/generate_random_instances.py \
    -n 20 \
    -p 0.05 \
    -s 42 \
    -o ../benchmarks/secondary
```

This populates `benchmarks/primary/` and `benchmarks/secondary/` with `.mzn`
instance files. The exact `(n, k)` and `(n, p)` grids used in the thesis are
listed in Table 1 (geographic) and §4.1 (random) of the report.

**Random seeds:** every instance is fully reproducible from a single base
seed. The generator takes a base seed via `-s`/`--seed`. When generating
`--count N` instances in one invocation, instance `idx` (0-indexed) uses
`seed = base_seed + idx` — so generating 30 instances with `--seed 42
--count 30` produces seeds 42 through 71. The exact seed used for each
instance is written directly into that instance's `.mzn` file as a header
comment (`%% seed = ...`), along with its `knn_edges` / `walk_edges` /
`walk_fraction` diagnostics — no separate seed log is needed, the seed is
self-documented per file.

For the thesis benchmarks, base seed `42` was used for both generators.

### 2. Build the Solver Variants

For each variant (e.g. V1):

```bash
cd solvers/pumpkin-v1
cargo build --release -p pumpkin-solver
```

Repeat for `pumpkin-v0`, `pumpkin-v2`, `pumpkin-v3`.

Before building, give each variant's MiniZinc solver configuration a unique
id, so MiniZinc can tell the four variants apart when flattening and
dispatching runs. Edit the `id` field in each variant's
`minizinc/pumpkin.msc` to something like
`nl.tudelft.algorithmics.pumpkin-circuit-v0` for V0,
`nl.tudelft.algorithmics.pumpkin-circuit-v1` for V1, and so on.

### 3. Flatten Instances & Build the Instance Database

`.mzn` instances must first be flattened to `.fzn` **per solver variant**
(each variant ships its own MiniZinc library/redefinitions), then indexed
into a single metadata file used by every later step.

```bash
# Flatten .mzn -> .fzn for one variant.
# Second argument is the MiniZinc solver id set in that variant's pumpkin.msc (step 2).
uv run python flatten.py pumpkin-v0 nl.tudelft.algorithmics.pumpkin-circuit-v0
# repeat for pumpkin-v1, pumpkin-v2, pumpkin-v3 with their respective solver ids
# output expected under flattened/pumpkinv0/, flattened/pumpkinv1/, etc.

# Validate that all 4 variants were flattened against the same instance set,
# and build the combined instances.csv metadata file (+ optional rewritten
# copies with output_var annotations for objective logging)
uv run python build-instance-database.py flattened/ --processed_dir processed/
```

`build-instance-database.py`:
- checks every variant's `flattened/<solver>/` folder contains exactly the
  same instance files (by stem), failing loudly if any variant is missing
  an instance
- extracts per-instance metadata (variable/constraint counts, objective
  name, `satisfy`/`minimize`/`maximize` type) into `flattened/instances.csv`
- if `--processed_dir` is given, writes a copy of each instance with its
  objective variable annotated `:: output_var`, so solver runs log the
  objective value at each improving solution (used for the common-bound
  comparison in [step 6](#6-aggregate-statistics))

### 4. Generate & Submit Experiments

```bash
# Generate SLURM job scripts + commands for one variant, e.g. V0,
# at the primary-benchmark 1800s timeout
uv run python generate-experiment.py flattened/instances.csv 1800 --solver pumpkin-v0

# Or generate for all 4 variants in one call by omitting --solver:
uv run python generate-experiment.py flattened/instances.csv 1800

# Submit the generated jobs
bash experiments/<timestamp>-pumpkin-v0/submit_jobs.sh
# or pass --submit to generate-experiment.py to submit automatically
```

Each run creates a timestamped folder
`experiments/<YYYYMMDD-HH.MM.SS.ffffff>-<solver>/` containing:
- `experiment.toml` — **automatically records the exact git SHA and branch**
  of the solver variant used for this run (read directly from
  `solvers/pumpkin-vX/`), giving an unambiguous, per-run provenance record
  independent of `SOLVER_VERSIONS.md`
- `commands.txt` — one `minizinc --solver ... --time-limit ...` invocation
  per instance
- `array_*.job` — SLURM array job script(s), split into chunks of at most
  1000 jobs per array
- `submit_jobs.sh` — submits all array jobs for this experiment
- one subfolder per instance run, containing `driver.log` (timing),
  `output.log` (solver stdout), `output.err` (solver stderr)

After running, **rename the timestamped folder** to its semantic experiment
name (see [Experiment Naming Scheme](#experiment-naming-scheme)), e.g.:

```bash
mv experiments/20260602-11.16.13.665988-pumpkin-v0 experiments/pri_fixed_pumpkin-v0
```

#### Cluster Details

Experiments were run on [DelftBlue](https://www.tudelft.nl/dhpc/) Phase 2,
via SLURM, with: 1 task, 1 CPU core per task, 4000 MB memory per CPU,
account `education-eemcs-courses-cse3000`, partition `compute-p2`. Per-job
wall-clock time limit is set dynamically from the `timeout_in_seconds`
argument to `generate-experiment.py` (1800s for the primary/geographic
benchmark, 300s for the secondary/random benchmark).

### 5. Parse Logs

Each instance run's `driver.log` / `output.log` / `output.err` must be
parsed into a `stats.toml` file before aggregation. There are **two
separate scripts**, both taking the same arguments — one parses only
external/runtime statistics, the other additionally parses
variant-specific internal solver statistics (propagation/clause-quality
data, which is only meaningful once you also need things like LBD or
pruning rate):

```bash
# External statistics only - ones mentioned by flag
uv run python parse-logs-no-internal-stats.py \
    ./flattened/instances.csv \
    experiments \
    1800 \
    --all \
    --statistic nodes \
    --statistic failures \
    --statistic propagations \
    --statistic propagators \
    --statistic solveTime \
    --statistic peakDepth \
    --statistic AverageConflictSize \
    --statistic AverageLbd \
    --statistic AverageLearnedNogoodLength \
    --statistic NumUnitNogoodsLearned \
    --statistic AverageBacktrackAmount

# Same as above but also parses independent variant-specific stats
uv run python parse-logs-internal-stats.py \
    ./flattened/instances.csv \
    experiments \
    1800 \
    --all \
    --statistic nodes \
    --statistic failures \
    --statistic propagations \
    --statistic propagators \
    --statistic solveTime \
    --statistic peakDepth \
    --statistic AverageConflictSize \
    --statistic AverageLbd \
    --statistic AverageLearnedNogoodLength \
    --statistic NumUnitNogoodsLearned \
    --statistic AverageBacktrackAmount
```

Positional arguments are `instances.csv`, the `experiments` directory, and
the timeout in seconds; `--all` parses every experiment subfolder, and
`--statistic <name>` is repeated once per statistic to extract.

### 6. Aggregate Statistics

```bash
# Aggregate one experiment folder
uv run python aggregate-statistics.py \
    flattened/instances.csv \
    experiments/pri_fixed_pumpkin-v0

# Aggregate every experiment subfolder under experiments/ in one call
uv run python aggregate-statistics.py \
    flattened/instances.csv \
    experiments/ \
    --all

# Optionally also compute statistics restricted to the best bound common to
# all variants (useful for comparing solution quality at a fixed objective)
uv run python aggregate-statistics.py \
    flattened/instances.csv \
    experiments/ \
    --all \
    --common_bound
```

For each experiment folder, this reads every run's `stats.toml`, joins it
against `instances.csv`, and writes:
- `statistics.json` — full per-run data
- `statistics_summary.csv` — one row per run, with `best_sol` computed from
  the recorded solution objectives
- `statistics_summary_common_bounds.csv` — only if `--common_bound` is set

**Manual step:** copy/rename each experiment's `statistics_summary.csv` into
`raw_statistics/<exp_family>/statistics-vX.csv`, e.g.:

```bash
cp experiments/pri_fixed_pumpkin-v0/statistics_summary.csv raw_statistics/pri_fixed/statistics-v0.csv
```

This is a manual step (no helper script).

### 7. Generate Tables & Figures

```bash
uv run python experiment-analysis/analyse_results_primary.py \
    --v0 raw_statistics/pri_fixed/statistics-v0.csv \
    --v1 raw_statistics/pri_fixed/statistics-v1.csv \
    --v2 raw_statistics/pri_fixed/statistics-v2.csv \
    --v3 raw_statistics/pri_fixed/statistics-v3.csv \
    --timeout 1800

uv run python experiment-analysis/analyse_results_sec.py \
    --v0 raw_statistics/sec_fixed/statistics-v0.csv \
    --v1 raw_statistics/sec_fixed/statistics-v1.csv \
    --v2 raw_statistics/sec_fixed/statistics-v2.csv \
    --v3 raw_statistics/sec_fixed/statistics-v3.csv \
    --timeout 300
```

`--vN` points to that variant's aggregated `statistics-vN.csv`;
`--timeout` is the per-instance timeout **in seconds**, used for PAR-2
scoring (penalises a timed-out instance at `2 × timeout`).

Each run produces, under `experiment-analysis/results/<experiment>/`:
- median/IQR runtime, solve-rate, and PAR-2 tables and plots, per variant
- internal-statistics plots (pruning rate, SCC size, LBD, nogood length,
  Hall-circuit fire rate, etc.)
- `report_figures/` — the exact figures embedded in the thesis report

**VSIDS instances (Exp 1.2):** these reuse the same `analyse_results_primary.py`
script, pointed at `raw_statistics/pri_vsids/` instead of `pri_fixed/`. The
VSIDS variant of the benchmark suite is produced by removing the explicit
search annotation line (`solve :: int_search(succ, input_order,
indomain_min) minimize maxleg;`) from the `.mzn` files before flattening,
which falls back to the solver's default activity-based (VSIDS) search.


---

## Experiment Naming Scheme

Experiment folders under `experiments/` follow:

```
<benchmark>_<search>_<variant>
```

| Part | Values | Meaning |
|---|---|---|
| `<benchmark>` | `pri` / `sec` | `pri` = primary, geographic k-nearest benchmark; `sec` = secondary, Erdős–Rényi random benchmark |
| `<search>` | `fixed` / `vsids` | `fixed` = input-order + indomain-min; `vsids` = activity-based VSIDS branching |
| `<variant>` | `pumpkin-v0` / `pumpkin-v1` / `pumpkin-v2` / `pumpkin-v3` | Propagator variant, see [Solver Variants](#solver-variants) |

This gives 12 experiment folders in total: `pri_fixed_pumpkin-v{0..3}`,
`pri_vsids_pumpkin-v{0..3}`, `sec_fixed_pumpkin-v{0..3}` — matching the 3
experiments × 4 variants described in the thesis. Note: folders are
generated with a timestamp prefix by `generate-experiment.py` and renamed
to this scheme afterward (see [step 4](#4-generate--submit-experiments)).
Under `raw_statistics/`, the per-variant CSVs use the shorter `statistics-vN.csv`
form, since they are already nested inside a folder named for the benchmark
and search strategy.

---

## Experiment ↔ Thesis Section Mapping

| Thesis section | Research question | Experiment folders |
|---|---|---|
| §4.3 Experiment 1.1 | RQ1: Does matching-based propagation help, and which variant wins? | `experiments/pri_fixed_pumpkin-v{0..3}/` |
| §4.4 Experiment 1.2 | RQ2: Do rankings hold under VSIDS search? | `experiments/pri_vsids_pumpkin-v{0..3}/` |
| §4.5 Experiment 2 | RQ3: Do effects generalise to unstructured graphs? | `experiments/sec_fixed_pumpkin-v{0..3}/` |

---

## Provenance

All solver variants under `solvers/` are squashed imports from the live
development fork (`git subtree`, see [`SOLVER_VERSIONS.md`](./SOLVER_VERSIONS.md)
for the commit each folder was imported from).

In addition, **every individual experiment run records its own provenance
automatically**: `generate-experiment.py` writes an `experiment.toml` file
into each experiment folder containing the exact git SHA and branch name of
the solver variant at the time the experiment was generated. This means
results can always be traced back to an exact version of the propagator
code — independent of, and as a cross-check against, `SOLVER_VERSIONS.md` —
even as development on the fork continues after thesis submission.