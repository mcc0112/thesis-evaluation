#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import subprocess


BENCHMARK_DIR = Path("benchmarks/random")
OUT_DIR = Path("flattened")


def flatten(mzn_lib: Path, out_dir: Path, model: Path, data: Path | None = None):
    mzn_input = [model] if data is None else [model, data]

    out_name = f"{model.stem}.fzn" if data is None else f"{model.stem}_{data.stem}.fzn"
    out_file = out_dir / out_name

    if out_file.exists():
        return

    args = ["minizinc", "-c", "--solver", mzn_lib, "--fzn", out_file] + mzn_input
    completed_process = subprocess.run(args)

    if completed_process.returncode != 0:
        error_msg = f"Failed to flatten {model.stem}"

        if data is not None:
            error_msg = f"{error_msg} with {data.stem}"

        print(error_msg)

    assert completed_process.returncode == 0


def run(mzn_lib_name: str, mzn_lib: Path):
    for challenge_dir in BENCHMARK_DIR.iterdir():
        if not challenge_dir.is_dir():
            continue

        print(f"Flattening {challenge_dir.name}")
        challenge_out = OUT_DIR / mzn_lib_name / challenge_dir.name 

        for family_dir in challenge_dir.iterdir():
            if not family_dir.is_dir():
                continue

            if family_dir.name.endswith(".IGNORED"):
                continue

            print(f"  Flattening {family_dir.name}")
            family_out_dir = challenge_out / family_dir.stem
            family_out_dir.mkdir(parents=True, exist_ok=True)

            for model_path in family_dir.rglob("*.mzn"):
                if model_path.stem.startswith("."):
                    continue

                has_data = (
                    False  # If there is no data files, then the MZN contains the data.
                )

                data_files = chain(
                    family_dir.rglob("*.dzn"), family_dir.rglob("*.json")
                )
                for data_path in data_files:
                    if data_path.stem.startswith("."):
                        continue

                    has_data = True
                    flatten(mzn_lib, family_out_dir, model_path, data_path)

                if not has_data:
                    flatten(mzn_lib, family_out_dir, model_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser("Flatten all the benchmarks with a given MZN library.")

    arg_parser.add_argument(
        "mzn_lib_name",
        type=Path,
        help="The name of this library. Used to separate the flatzinc files in subdirectories.",
    )
    arg_parser.add_argument(
        "mzn_lib", type=Path, help="The path to the minizinc library to use."
    )

    args = arg_parser.parse_args()

    run(args.mzn_lib_name, args.mzn_lib)
