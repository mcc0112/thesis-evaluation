#!/usr/bin/env python3

import os
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import tomli
from tqdm import tqdm


def get_default_value(value):
    if isinstance(value, str):
        return ""

    if isinstance(value, int):
        return 0

    if isinstance(value, float):
        return 0.0

    if isinstance(value, list):
        return []

    raise Exception(f"Cannot get default value for type {type(value)}")


def run(instances_path: Path, experiment: Path, aggregate_all: bool, common_bound: bool):
    instances = pl.read_csv(instances_path)

    if aggregate_all:
        directories = [Path(entry.path) for entry in os.scandir(experiment)]
    else:
        directories = [experiment]

    instance_to_common_bounds = {}

    for experiment in directories:
        if not experiment.is_dir():
            continue

        print(f"Aggregating {experiment}")

        aggregated_stats = {}
        idx = 0

        skipped = 0

        for item in tqdm(list(filter(lambda x: x.is_dir(), experiment.iterdir()))):
            stats_file_path = item / "stats.toml"
            if not stats_file_path.is_file():
                skipped += 1
                idx += 1  # still counts as a row
                continue

            with stats_file_path.open("rb") as f:
                stats = tomli.load(f)

            if common_bound and "instance" in stats and "solution_objectives" in stats:
                instance = stats["instance"]
                bounds = set([solution_info["value"] for solution_info in stats["solution_objectives"]])
                instance_type = instances.row(by_predicate=(pl.col("path") == instance), named=True)["type"]
                if instance not in instance_to_common_bounds:
                    instance_to_common_bounds[instance] = {"bounds": bounds, "type": instance_type}
                else:
                    instance_to_common_bounds[instance]["bounds"] &= bounds

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

            idx += 1  # increment at end of each item

        df = pl.DataFrame(aggregated_stats, strict=False).join(
            instances.select([pl.col("path").alias("instance"), pl.col("type")]),
            on="instance",
        )
        df.write_json(experiment / "statistics.json")

        obj = pl.col("solution_objectives").list.eval(
            pl.element().struct.rename_fields(["objective", "time"]).struct.field("objective")
        )

        summary = df.with_columns(
            best_sol=pl.when(pl.col("type") == "minimize")
            .then(obj.list.min())
            .otherwise(pl.when(pl.col("type") == "maximize").then(obj.list.max()).otherwise(None))
        ).select(pl.exclude("solution_objectives"))
        summary.write_csv(experiment / "statistics_summary.csv")

        print(f"Finished. Skipped {skipped} runs.")

    if common_bound:
        instance_to_best_bound = {}
        for instance, info in instance_to_common_bounds.items():
            if len(info["bounds"]) == 0:
                continue
            else:
                if info["type"] == "minimize":
                    instance_to_best_bound[instance] = min(info["bounds"])
                elif info["type"] == "maximize":
                    instance_to_best_bound[instance] = max(info["bounds"])

        print("Starting calculation of statistics at best common bound")
        for experiment in directories:
            if not experiment.is_dir():
                continue

            print(f"Aggregating {experiment}")

            aggregated_stats = {}
            idx = 0

            skipped = 0

            for item in tqdm(list(filter(lambda x: x.is_dir(), experiment.iterdir()))):
                stats_file_path = item / "stats.toml"
                if not stats_file_path.is_file():
                    skipped += 1
                    idx += 1
                    continue

                with stats_file_path.open("rb") as f:
                    stats = tomli.load(f)

                if (
                    "instance" in stats
                    and stats["instance"] in instance_to_best_bound
                    and "solution_objectives" in stats
                ):
                    stats["common_bound"] = instance_to_best_bound[stats["instance"]]
                    for solution_info in stats["solution_objectives"]:
                        if solution_info["value"] == instance_to_best_bound[stats["instance"]]:
                            best_sol = stats["solution_objectives"][-1]["value"]
                            if "best_sol" not in aggregated_stats:
                                default_value = get_default_value(best_sol)
                                aggregated_stats["best_sol"] = [default_value] * idx

                            aggregated_stats["best_sol"].append(best_sol)

                            for key, value in stats.items():
                                if key not in solution_info:
                                    if key not in aggregated_stats:
                                        default_value = get_default_value(value)
                                        aggregated_stats[key] = [default_value] * idx

                                    aggregated_stats[key].append(value)

                            for key, value in solution_info.items():
                                if key not in aggregated_stats:
                                    default_value = get_default_value(value)
                                    aggregated_stats[key] = [default_value] * idx

                                aggregated_stats[key].append(value)
                            break

                for key, value in aggregated_stats.items():
                    if len(value) > idx:
                        continue

                    assert len(value) > 0

                    default_value = get_default_value(value[0])
                    value.append(default_value)

                idx += 1

            df = pl.DataFrame(aggregated_stats, strict=False).join(
                instances.select([pl.col("path").alias("instance"), pl.col("type")]),
                on="instance",
            )
            obj = pl.col("solution_objectives").list.eval(
                pl.element().struct.rename_fields(["objective", "time"]).struct.field("objective")
            )

            summary = df.select(pl.exclude("solution_objectives"))
            summary.write_csv(experiment / "statistics_summary_common_bounds.csv")

            print(f"Finished common bounds. Skipped {skipped} runs.")


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("instances", type=Path)
    arg_parser.add_argument("experiment", type=Path)
    arg_parser.add_argument(
        "--all", action="store_true", help="If set to true then all subfolders in `experiment` are parsed"
    )
    arg_parser.add_argument(
        "--common_bound",
        action="store_true",
        help="If set to true then a statistic summary across the best common bound is created",
    )

    args = arg_parser.parse_args()

    run(args.instances, args.experiment, args.all, args.common_bound)