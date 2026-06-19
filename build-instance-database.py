#!/usr/bin/env python3

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import re
from typing import Optional

import polars as pl


# Find the objective variable name from a line that starts with 'solve'.
OBJECTIVE_NAME_REGEX = re.compile(r"([A-Za-z][A-Za-z0-9_]*);")

# Number of threads to use when parallelizing
THREAD_COUNT = 4


def extract_statistics(flattened_dir: Path, instance: Path) -> dict:
    num_variables = 0
    num_constraints = 0

    objective = "-"
    instance_type = "unknown"

    constraint_counts = {}

    with instance.open("r") as contents:
        contents = contents.read()
        for line in contents.splitlines():
            if line.startswith("var"):
                num_variables += 1

            if line.startswith("constraint"):
                num_constraints += 1

                constraint = line.removeprefix("constraint ")
                constraint_name, _, _ = constraint.partition("(")

                count = constraint_counts.get(constraint_name, 0)
                constraint_counts[constraint_name] = count + 1

            if line.startswith("solve"):
                if line.find("satisfy") >= 0:
                    instance_type = "satisfy"
                else:
                    if line.find("minimize") >= 0:
                        instance_type = "minimize"
                    elif line.find("maximize") >= 0:
                        instance_type = "maximize"
                    else:
                        raise Exception(
                            f"Cannot get optimisation direction from line '{line}'"
                        )

                    match_result = OBJECTIVE_NAME_REGEX.search(line)
                    if match_result is None:
                        raise Exception(f"Cannot get objective name from line '{line}'")

                    objective = match_result.group(1)

    instance_path = instance.relative_to(flattened_dir)
    solver = instance_path.parts[0]

    print(f"Finished parsing {instance_path}")

    return {
        "instance": instance.stem,
        "family": instance.parent.name,
        "solver": solver,
        "type": instance_type,
        "objective_name": objective,
        "path": str(instance_path),
        "num_variables": num_variables,
        "num_constraints": num_constraints,
        **constraint_counts,
    }


def get_default_value(value):
    if isinstance(value, str):
        return ""

    if isinstance(value, int):
        return 0

    raise Exception(f"Cannot get default value for type {type(value)}")


def gather_instances(directory: Path) -> pl.DataFrame:
    aggregated_stats = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for instance in directory.rglob("*.fzn"):
            futures.append(
                executor.submit(
                    extract_statistics, flattened_dir=directory, instance=instance
                )
            )

        for idx, future in enumerate(as_completed(futures)):
            stats = future.result()

            for key, value in stats.items():
                if key not in aggregated_stats:
                    default_value = get_default_value(value)
                    aggregated_stats[key] = [default_value] * idx

                aggregated_stats[key].append(value)

            for key, value in aggregated_stats.items():
                if len(value) > idx:
                    continue

                assert len(value) > 0

                default_value = get_default_value(value[0])
                value.append(default_value)

    return pl.DataFrame(aggregated_stats)


def check_instances_match(representative: Path, other: Path) -> bool:
    assert representative.is_dir()
    assert other.is_dir()

    def stem_only(p: Path) -> str:
        return p.stem

    representative_contents = set(map(stem_only, representative.iterdir()))
    other_contents = set(map(stem_only, other.iterdir()))

    diff = representative_contents.symmetric_difference(other_contents)

    if len(diff) != 0:
        print("Mismatch between folders:")
        print(f"  - {representative}")
        print(f"  - {other}")
        print("")

        print("Unpaired items:")
        for item in diff:
            print(f"  - {item}")

        return False

    success = True

    directories = filter(
        lambda p: (representative / p).is_dir(), representative_contents
    )
    for sub_directory in directories:
        success = success and check_instances_match(
            representative / sub_directory, other / sub_directory
        )

    return success


def validate_instances(flattened_dir: Path) -> bool:
    solvers = [p for p in flattened_dir.iterdir() if p.is_dir()]

    if len(solvers) == 0:
        print("No solvers found.")
        return False

    print("Identified solvers:")
    for solver in solvers:
        print(f"  - {solver.stem}")
    print("")

    # We pick one of the solvers as a representative. Then we ensure that its
    # instances match the instances for the other solvers.

    representative = solvers.pop()

    success = True
    for other in solvers:
        success = success and check_instances_match(representative, other)

    return success


def process_instance(input_path: Path, output_path: Path, var_name_to_output: str):
    with input_path.open("r") as input_file:
        with output_path.open("w") as output_file:
            for line in input_file:
                is_variable = line.startswith("var")
                is_objective = var_name_to_output in line
                is_output = "output_var" not in line

                if is_variable and is_objective and is_output:
                    prefix, _, _ = line.partition(";")

                    output_file.write(f"{prefix} :: output_var;\n")
                else:
                    output_file.write(f"{line}")


def process_instances(df: pl.DataFrame, input_dir: Path, output_dir: Path):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for row in df.iter_rows(named=True):
            input_file_path: Path = input_dir / row["path"]
            output_file_path: Path = output_dir / row["path"]

            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if row["type"] == "satisfy":
                shutil.copy(input_file_path, output_file_path)
            else:
                assert row["objective_name"] != "-"

                futures.append(
                    executor.submit(
                        process_instance,
                        input_path=input_file_path,
                        output_path=output_file_path,
                        var_name_to_output=row["objective_name"],
                    )
                )


def run(flattened_dir: Path, processed_dir: Path | None):
    if not validate_instances(flattened_dir):
        sys.exit(1)

    df = gather_instances(flattened_dir)

    print(f"Parsed {len(df)} instances.")

    csv_path = flattened_dir / "instances.csv"
    df.write_csv(csv_path)
    print(f"Written instances to {csv_path}")

    if processed_dir:
        print(f"Rewriting to log objectives...")
        process_instances(df, flattened_dir, processed_dir)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Checks that multiple solvers will run on the same instances."
    )

    arg_parser.add_argument(
        "flattened_dir",
        type=Path,
        help="The directory containing the flattened instances for all solvers.",
    )

    arg_parser.add_argument(
        "--processed_dir",
        type=Path,
        help="The directory containing the processed flattened instances for all solvers.",
    )

    args = arg_parser.parse_args()
    run(args.flattened_dir, args.processed_dir)
