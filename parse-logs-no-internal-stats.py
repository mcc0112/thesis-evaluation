#!/usr/bin/env python3
import os
import pprint
import re
from argparse import ArgumentParser
from collections import deque
from functools import reduce
from os.path import isfile
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import tomli
import tomli_w
from tqdm import tqdm

from common import CONFIG_TO_SOLVER

CONFLICT_COUNT_PREFIX = "failures="
TIME_ELAPSED_PREFIX = "% time elapsed:"
SOLUTION_MARKER = "-" * 10

TRIM_PARTS_START = 5


def split(string, sep, n) -> Tuple[str, str]:
    """Split `string´ at the `n`th occurrence of `sep`"""
    pos = reduce(lambda x, _: string.index(sep, x + 1), range(n + 1), -1)
    return string[:pos], string[pos + len(sep) :]


def clean_line(line: str) -> str:
    result = []
    line = line.replace("'", '"')
    number_of_quotes = line.count('"')
    if number_of_quotes % 2 != 0:
        line += '"'
    if number_of_quotes == 0:
        return line

    occurrence_quote = 0
    for character in line:
        if character == '"':
            if occurrence_quote > 0 and occurrence_quote < number_of_quotes:
                result.append("'")
            else:
                result.append(character)
            occurrence_quote += 1
        else:
            result.append(character)

    return "".join(result)


def find_statistic_and_add_to_dict(log: str, statistic_name: str, statistics: Dict[str, str]):
    stat_regex = re.compile(rf"%%%mzn-stat: {statistic_name}=([0-9]+\.?[0-9]*)")
    result = deque(stat_regex.finditer(log), 1)
    if len(result) == 0:
        stat_value = "-"
    else:
        last = result.pop()
        stat_value = last.group(1)

    statistics[statistic_name] = stat_value


def run(
    instances_path: Path,
    experiment: Path,
    parse_all: bool,
    check: bool,
    timeout: int,
    print_error_info: bool,
    statistics: List[str],
):
    instances = pl.read_csv(instances_path)

    if check:
        instances_to_best_solutions = {}

    if parse_all:
        directories = [Path(entry.path) for entry in os.scandir(experiment)]
    else:
        directories = [experiment]

    for experiment in directories:
        if not experiment.is_dir():
            continue
        _, solver_config = split(experiment.name, "-", 1)
        solver = CONFIG_TO_SOLVER[solver_config]

        print(f"Processing {solver} in {experiment}")

        instances_solver = instances.filter(pl.col("solver") == solver)

        overview = {}

        items = 0
        missing_logs = 0
        out_of_memory = 0

        for item in tqdm(list(filter(lambda x: x.is_dir(), experiment.iterdir()))):
            items += 1
            driver_log_path = item / "driver.log"
            if not driver_log_path.is_file():
                missing_logs += 1
                continue

            with driver_log_path.open("rb") as driver_log:
                content = driver_log.read().decode("utf-8")
                cleaned = "\n".join(
                    [
                        (
                            line[: equality_index + 1] + clean_line(line[equality_index + 1 : -1])
                            if '"' in line
                            else line
                        )
                        for line in content.split("\n")
                        if (equality_index := content.find("=")) and len(line) > 0
                    ]
                )
                driver: dict[str, str] = tomli.loads(cleaned)

            if "command" in driver:
                instance = Path(driver["command"].split(" ")[-1])
                instance = Path(*instance.parts[TRIM_PARTS_START:])

                if len(instance.parts) == 0:
                    print("================================")
                    print("EMPTY INSTANCE PATH")
                    print("Experiment:", experiment)
                    print("Run directory:", item)
                    print("Driver contents:", driver)
                    raise RuntimeError("Empty instance path")

                assert instance.parts[0] == solver, (
                    f"Expected the first component '{instance.parts[0]}' in the instance path to be the solver name"
                    f" '{solver}'."
                )
            else:
                assert "file" in driver
                instance = Path(
                    "/scratch/icwmmarijnisse/pumpkin-revamp-paper-evaluation/processed/pumpkin/"
                    + driver["file"].split(", ")[1].removeprefix("../../../").removesuffix("]")
                )
                instance = Path(*instance.parts[TRIM_PARTS_START:])

                assert instance.parts[0] == solver, (
                    f"Expected the first component '{instance.parts[0]}' in the instance path to be the solver name"
                    f" '{solver}'."
                )

            key = instance.stem

            try:
                instance_meta = instances_solver.row(by_predicate=(pl.col("path") == str(instance)), named=True)
            except pl.exceptions.NoRowsReturnedError as e:
                tqdm.write(f"Failed to find metadata for {key} on {solver}")
                raise e

            status = "UNKNOWN"

            solver_log_path = item / "output.log"
            if not solver_log_path.is_file():
                solver_log_path = item / "run.log"
                if not solver_log_path.is_file():
                    missing_logs += 1
                    continue

            with solver_log_path.open("r") as solver_log:
                log = solver_log.read()

            raw_solutions = log.split(SOLUTION_MARKER)
            raw_solutions.pop()

            solutions = list(
                filter(
                    lambda s: len(s) > 0 and not "=" * 5 in s,
                    map(lambda s: s.strip(), raw_solutions),
                )
            )

            status = "UNKNOWN"
            if "command_status" not in driver and "command" in driver:
                if len(solutions) > 0:
                    status = "SATISFIABLE"
                else:
                    status = "UNKNOWN"
            elif "command_status" in driver and int(driver["command_status"]) != 0:
                status = "ERROR"

                err_path = item / "output.err"

                if os.path.isfile(err_path):
                    with open(err_path, "r") as err_file:
                        lines = err_file.readlines()
                        if sum([1 for line in lines if "Out Of Memory" in line]) > 0:
                            out_of_memory += 1
                            status = "ERROR"
                        else:
                            if print_error_info:
                                joined_lines = "\n".join(lines)
                                tqdm.write(
                                    f"==============================FOUND UNKNOWN ERROR in {item} for"
                                    f" {instance} ==============================\n{joined_lines}\n"
                                )
            elif "UNSATISFIABLE" in log:
                status = "UNSATISFIABLE"
            elif "=" * 10 in log:
                if instance_meta["type"] == "satisfy":
                    status = "SATISFIABLE_COMPLETE_ENUMERATION"
                else:
                    assert instance_meta["type"] in ["minimize", "maximize"]
                    status = "OPTIMAL"
            elif len(solutions) > 0:
                status = "SATISFIABLE"

            if status not in overview:
                overview[status] = 0
            overview[status] += 1

            solution_objectives = []

            # For a satisfaction problem we don't record the solutions. For
            # optimisation problems, we need every reported solution along with
            # the time it took to discover it.
            if status != "ERROR" and instance_meta["type"] != "satisfy":
                objective_var = instance_meta["objective_name"]
                objective_prefix = f"{objective_var} = "

                for solution in solutions:
                    objective_line = None
                    time_line = None

                    solution_info = {}

                    for l in solution.splitlines():
                        if l.startswith(objective_prefix) and l.endswith(";"):
                            objective_line = l
                        if l.startswith(TIME_ELAPSED_PREFIX) and l.endswith(" s"):
                            time_line = l

                        for statistic in statistics:
                            statistic_prefix = f"%%%mzn-stat: {statistic}="
                            if l.startswith(statistic_prefix):
                                statistic_value = l.removeprefix(statistic_prefix)
                                try:
                                    statistic_value = int(statistic_value)
                                except ValueError:
                                    statistic_value = float(statistic_value)

                                solution_info[statistic] = statistic_value

                                break

                    assert objective_line is not None and time_line is not None

                    objective_value = int(objective_line.removeprefix(objective_prefix).removesuffix(";"))
                    solution_info["value"] = objective_value
                    time_spent = float(time_line.removeprefix(TIME_ELAPSED_PREFIX).removesuffix(" s"))
                    solution_info["time"] = time_spent

                    solution_objectives.append(solution_info)

                assert len(solution_objectives) == len(solutions)

            if check:
                if str(instance) not in instances_to_best_solutions:
                    instances_to_best_solutions[str(instance)] = {
                        "instance_type": instance_meta["type"],
                        "objectives_and_statuses": [],
                    }
                if len(solution_objectives) > 0:
                    instances_to_best_solutions[str(instance)]["objectives_and_statuses"].append(
                        {
                            "status": status,
                            "best_objective": solution_objectives[-1]["value"],
                            "solver": experiment.name,
                        }
                    )
                if status == "UNSATISFIABLE":
                    instances_to_best_solutions[str(instance)]["objectives_and_statuses"].append(
                        {
                            "status": status,
                            "best_objective": None,
                            "solver": experiment.name,
                        }
                    )

            wall_clock_time = float(driver["wall_clock_time"]) if "wall_clock_time" in driver else float(timeout)
            if "cpu_user_time" in driver:
                assert "cpu_system_time" in driver
                cpu_user_time = float(driver["cpu_user_time"]) + float(driver["cpu_system_time"])
            elif "user_time" in driver:
                assert "system_time" in driver
                cpu_user_time = float(driver["user_time"]) + float(driver["system_time"])
            else:
                cpu_user_time = float(timeout)

            stats = {
                "instance": str(instance),
                "directory": str(item),
                "status": status,
                "num_solutions": len(solutions),
                "wall_clock_time": wall_clock_time,
                "cpu_time": cpu_user_time,
                "solution_objectives": solution_objectives,
            }

            for statistic in statistics:
                find_statistic_and_add_to_dict(log, statistic, stats)

            stats_file_path = item / "stats.toml"
            with stats_file_path.open("wb") as stats_file:
                tomli_w.dump(stats, stats_file)

        pprint.pprint(overview)
        print(f"Skipped {missing_logs} from {items} runs.")
        print(f"Detected OOM in {out_of_memory} from {items} runs.")

    if check:
        for instance, info in instances_to_best_solutions.items():
            instance_type = info["instance_type"]
            objectives_and_statuses = info["objectives_and_statuses"]

            best_objective = None
            unsatisfiable = False
            for objective_and_status in objectives_and_statuses:
                status = objective_and_status["status"]
                if status == "UNSATISFIABLE":
                    unsatisfiable = True
                    continue
                objective = objective_and_status["best_objective"]

                if instance_type == "maximize":
                    if best_objective is None:
                        best_objective = objective
                    else:
                        best_objective = max(best_objective, objective)
                elif instance_type == "minimize":
                    if best_objective is None:
                        best_objective = objective
                    else:
                        best_objective = min(best_objective, objective)

            for objective_and_status in objectives_and_statuses:
                status = objective_and_status["status"]
                objective = objective_and_status["best_objective"]

                if unsatisfiable and objective is not None:
                    print(
                        f"Expected unsatisfiable but solution with objective {objective} was found for instance"
                        f" {instance}\n{info}"
                    )

                if status == "OPTIMAL" and objective != best_objective:
                    print(
                        f"Expected reported optimal solution {objective} to have the same cost as the best found"
                        f" {best_objective} for instance {instance}\n{info}"
                    )
                elif status == "SATISFIABLE":
                    if instance_type == "maximize":
                        if objective > best_objective:
                            print(
                                f"Expected reported satisfiable solution {objective} to not have a better objective"
                                f" than the best found found {best_objective} for instance {instance}\n{info}"
                            )

                    elif instance_type == "minimize":
                        if objective < best_objective:
                            print(
                                f"Expected reported satisfiable solution {objective} to not have a better objective"
                                f" than the best found found {best_objective} for instance {instance}\n{info}"
                            )


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("instances", type=Path)
    arg_parser.add_argument("experiment", type=Path)
    arg_parser.add_argument(
        "timeout",
        type=int,
        help=(
            "The timeout in seconds which was used; this is used to determine the time in case the run did not complete"
            " and did not write to the driver file"
        ),
    )
    arg_parser.add_argument("--statistic", action="append")
    arg_parser.add_argument(
        "--all", action="store_true", help="If set to true then all subfolders in `experiment` are parsed"
    )
    arg_parser.add_argument(
        "--no-check",
        action="store_true",
        help="If set to true then a simple check whether the bounds reported are consistent is not performed",
    )
    arg_parser.add_argument(
        "--no-errors",
        action="store_true",
        help="If set to true, then it will not print more information related to 'ERROR' instances.",
    )

    args = arg_parser.parse_args()

    run(
        args.instances,
        args.experiment,
        args.all,
        not args.no_check,
        args.timeout,
        not args.no_errors,
        args.statistic if args.statistic else [],
    )
