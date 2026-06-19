#!/usr/bin/env python3

import os
import stat
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from math import ceil, floor, log10

import polars as pl

from common import *
import re
TRIM_PARTS_START = 5

SOLVER_SRC = Path("./pumpkin").resolve()
SOLVER_PATH = SOLVER_SRC / "target" / "rel-with-debug" / "pumpkin-solver"

# Time VTune has to clean up in seconds
VTUNE_CLEANUP_TIME = 2 * 60
# Memory in GB
MEMORY = 4000

# The maximum size of a job array.
MAX_ARRAY_SIZE = 1000

SBATCH_FILE_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name=@JOB_NAME@
#SBATCH --array=1-@NUM_JOBS@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --time=@TIME@
#SBATCH --mem-per-cpu=@MEMORY@M

#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=compute-p2


set -euo pipefail
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
source $HOME/.cargo/env
export PATH=/scratch/mcarrio/MiniZincIDE-2.8.3-bundle-linux-x86_64/bin:$PATH
module load miniforge3
conda activate pumpkin-eval

# How far to offset the SLURM_ARRAY_TASK_ID into the commands file.
#
# Required because multiple arrays may be necessary to execute all the commands.
ID_OFFSET=@ID_OFFSET@

COMMAND_ID=$(($ID_OFFSET+$SLURM_ARRAY_TASK_ID))

# Read the command from the commands file. This file lists all the commands on 
# separate lines, and we use $COMMAND_ID to find the line and extract
# the command.
COMMAND=$(sed "${COMMAND_ID}q;d" @COMMANDS_FILE@)
# Expand into array while preserving quoting
eval "COMMAND_ARRAY=($COMMAND)"

RUN_DIR=@RUN_DIR_PREFIX@/$COMMAND_ID
mkdir -p $RUN_DIR

DRIVER_LOG="$RUN_DIR/driver.log"
OUTPUT_LOG="$RUN_DIR/output.log"
ERROR_LOG="$RUN_DIR/output.err"

# Put the command into the driver log.
echo "command = \\\"${COMMAND}\\\"" > $DRIVER_LOG

# Runs the command with exact argument separation
srun /usr/bin/time \\
        --quiet \\
        --format="command_status = %x\\nwall_clock_time = %e\\ncpu_user_time = %U\\ncpu_system_time = %S" \\
        --append \\
        --output=$DRIVER_LOG \\
         "${COMMAND_ARRAY[@]}" > $OUTPUT_LOG 2> $ERROR_LOG
""".lstrip()

VTUNE_SBATCH_FILE_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name=@JOB_NAME@
#SBATCH --array=1-@NUM_JOBS@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --time=@TIME@
#SBATCH --mem-per-cpu=@MEMORY@M

#SBATCH --account=research-eemcs-st
#SBATCH --partition=compute-p2

#SBATCH --output=@SLURM_OUTPUT@/%j.out
#SBATCH --error=@SLURM_OUTPUT@/%j.err

module load 2023r1-intel

# How far to offset the SLURM_ARRAY_TASK_ID into the commands file.
#
# Required because multiple arrays may be necessary to execute all the commands.
ID_OFFSET=@ID_OFFSET@

COMMAND_ID=$(($ID_OFFSET+$SLURM_ARRAY_TASK_ID))

# Read the command from the commands file. This file lists all the commands on 
# separate lines, and we use $COMMAND_ID to find the line and extract
# the command.
COMMAND=$(sed "${COMMAND_ID}q;d" @COMMANDS_FILE@)
# Expand into array while preserving quoting
eval "COMMAND_ARRAY=($COMMAND)"

RUN_DIR_NAME=$(printf "%0@RUN_DIR_PADDING@d" $COMMAND_ID)

RUN_DIR=@RUN_DIR_PREFIX@/$RUN_DIR_NAME
mkdir -p $RUN_DIR
cd $RUN_DIR

DRIVER_LOG="$RUN_DIR/driver.log"
OUTPUT_LOG="$RUN_DIR/output.log"
ERROR_LOG="$RUN_DIR/output.err"

# Put the command into the driver log.
echo "command = \\\"${COMMAND}\\\"" > $DRIVER_LOG

srun /usr/bin/time \\
        --quiet \\
        --format="command_status = %x\\nwall_clock_time = %e\\ncpu_user_time = %U\\ncpu_system_time = %S" \\
        --append \\
        --output=$DRIVER_LOG \\
         "${COMMAND_ARRAY[@]}" > $OUTPUT_LOG 2> $ERROR_LOG
""".lstrip()

def run(
    solver: str, instances_path: Path, timeout: int, profiling=False, solver_path=None, should_submit: bool = False
):
    instances = pl.read_csv(instances_path)
    instances_solver = instances.filter(pl.col("solver") == solver)
    additional_flags = SOLVER_TO_FLAGS[solver] if solver in SOLVER_TO_FLAGS else ""

    if solver in SOLVER_TO_FLAGS:
        print(f"WARNING: Running with flags {additional_flags}")

    print(f"Generating experiment for {solver}.")

    # Create a new directory containing the logs.
    timestamp = datetime.now().strftime("%Y%m%d-%H.%M.%S.%f")
    experiment_dir = EXPERIMENTS_DIR_PREFIX / f"{timestamp}-{solver}"
    experiment_dir.mkdir(parents=True)

    metadata_file = experiment_dir / "experiment.toml"
    with open(metadata_file, "w+") as metadata:
        assert solver in SOLVER_DIRS, f"Could not find {solver} in SOLVER_DIRS; please update 'common.py'"
        print(f"Writing metadata to {metadata_file}")
        SHA = subprocess.run(
            ["git", "-C", f"{SOLVER_DIRS[solver]}", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout
        metadata.write(f"SHA = {SHA}")

        branch_name = subprocess.run(
            ["git", "-C", f"{SOLVER_DIRS[solver]}", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout
        metadata.write(f"branch = {branch_name}")
        metadata.write(f"flags = '{additional_flags}'")

    slurm_output_dir = experiment_dir / "slurm_output"
    slurm_output_dir.mkdir()

    # Generate a file with each of the commands on a new line.
    commands_file = experiment_dir / "commands.txt"
    num_commands = 0
    print(f"Generating experiment for {solver}.")

    print("INSTANCES path:", INSTANCES[solver])
    print("num files:", len(list(INSTANCES[solver].rglob('*.fzn'))))

    print("INSTANCES dict entry:", INSTANCES[solver])
    print("Rglob count:", len(list(INSTANCES[solver].rglob('*.fzn'))))
    print("TYPE:", type(INSTANCES[solver]))
    files = list(INSTANCES[solver].rglob("*.fzn"))
    print("FORCED LIST LENGTH:", len(files))
    with commands_file.open("w") as commands:
        for instance in files:
            solver_name = CONFIG_TO_SOLVER[solver]
            trimmed_instance = Path(*instance.parts[TRIM_PARTS_START:])
            instances_solver = instances.filter(pl.col("solver") == solver_name)
            key = trimmed_instance.stem

            try:
                instance_meta = instances_solver.row(by_predicate=(pl.col("path") == str(trimmed_instance)), named=True)
            except pl.exceptions.NoRowsReturnedError as e:
                print(f"Failed to find metadata for {key} on {solver}")
                raise e

            is_satisfaction_problem = instance_meta["type"] == "satisfy"

            num_commands += 1

            # Create run folder. This is necessary so we can pipe SLURM stdout and stderror to the correct place.
            if not profiling:
                (experiment_dir / str(num_commands)).mkdir(parents=True, exist_ok=True)

            free_search = "-f" if "-free" in solver else ""
            no_minimize = "--no-learning-minimise" if "-nominimization" in solver else ""
            learning = "--conflict-resolver decision" if "-decisionlearning" in solver else ""

            fzn_flags_array = list(filter(None, [additional_flags, learning, no_minimize, "-s", "-v"]))
            if not is_satisfaction_problem:
                fzn_flags_array.append("-a")
            fzn_flags = f"--backend-flags '{' '.join(fzn_flags_array)}'" if len(fzn_flags_array) > 0 else ""

            solver_msc = SOLVER_CONFIGS[solver]
            if profiling:
                SOLVER_SRC = Path(solver_path).resolve()
                SOLVER_PATH = SOLVER_SRC / "target" / "rel-with-debug" / "pumpkin-solver"
                if not os.path.isfile(SOLVER_PATH):
                    raise Exception("Target which is compiled with debug options does not exist")
                command = (
                    "vtune -collect hotspots -knob sampling-mode=sw -no-summary"
                    f" -search-dir={SOLVER_SRC / 'target'} -source-search-dir={SOLVER_SRC} -result-dir ./result --"
                    f" {SOLVER_PATH} -saf --time-limit {timeout * 1000} {instance}"
                )
            else:

                command = (
                    f"minizinc --time-limit {timeout * 1000} --output-time --solver"
                    f" {solver_msc} {free_search} {fzn_flags} {instance}"
                )
                command = command.strip()
                if not command:
                    raise ValueError(f"Empty command generated for instance {instance}")
            commands.write(f"{command}\n")

    # Create the submission files for slurm.
    num_job_arrays = ceil(num_commands / MAX_ARRAY_SIZE)

    for i in range(num_job_arrays):
        slurm_file = experiment_dir / f"array_{i + 1}.job"

        id_offset = i * MAX_ARRAY_SIZE
        num_jobs = min(MAX_ARRAY_SIZE, num_commands - id_offset)

        with slurm_file.open("w") as slurm:
            seconds = (timeout + VTUNE_CLEANUP_TIME) % (24 * 3600) if profiling else timeout % (24 * 3600)
            hour = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60

            input_sbatch = (
                VTUNE_SBATCH_FILE_TEMPLATE.replace("@SLURM_OUTPUT@", str(slurm_output_dir)).replace(
                    "@RUN_DIR_PADDING@", str(floor(log10(num_commands)) + 1)
                )
                if profiling
                else SBATCH_FILE_TEMPLATE
            )
            contents = (
                input_sbatch.replace("@ID_OFFSET@", str(id_offset))
                .replace("@RUN_DIR_PREFIX@", str(experiment_dir))
                .replace("@RUN_DIR_PREFIX@", str(experiment_dir))
                .replace("@JOB_NAME@", f"pumpkin-revamp-eval-{solver}")
                .replace("@COMMANDS_FILE@", str(commands_file))
                .replace("@TIME@", f"{hour:02d}:{minutes:02d}:{seconds:02d}")
                .replace("@MEMORY@", str(MEMORY))
                .replace("@NUM_JOBS@", str(num_jobs))
            )

            slurm.write(contents)

    submit_file = experiment_dir / "submit_jobs.sh"
    with submit_file.open("w") as submit:
        submit.write(f"#!/bin/bash\n\n")

        for i in range(num_job_arrays):
            submit.write(f"sbatch {experiment_dir}/array_{i + 1}.job\n")

    st = os.stat(submit_file)
    os.chmod(submit_file, st.st_mode | stat.S_IEXEC)

    if should_submit:
        os.chdir(experiment_dir)
        result = subprocess.check_output(["sh", submit_file], stderr=subprocess.STDOUT).decode()
        if not len(result) == 0:
            print(result)


if __name__ == "__main__":
    arg_parse = ArgumentParser()

    arg_parse.add_argument("instances", type=Path)
    arg_parse.add_argument("timeout_in_seconds", type=int)
    arg_parse.add_argument(
        "--solver",
        type=str,
        help="The solver to use for this experiment. If not specified, generate experiments for all configurations.",
        choices=CONFIGURATIONS,
    )
    arg_parse.add_argument(
        "--profiling",
        action="store_true",
        help="Whether to use vtune for profiling; note that this currently does not pass any of the flags to the run",
    )
    arg_parse.add_argument(
        "--solver_path",
        type=Path,
        help=(
            "When profiling is used, this contains the path to the root directory of the solver; currently only Pumpkin"
            " can be used"
        ),
    )
    arg_parse.add_argument(
        "--submit", action="store_true", help="When enabled, it will submit the jobs directly to the queue"
    )

    args = arg_parse.parse_args()

    if args.solver is not None:
        run(
            args.solver,
            args.instances,
            args.timeout_in_seconds,
            profiling=args.profiling,
            solver_path=args.solver_path,
            should_submit=args.submit,
        )
    else:
        for solver in CONFIGURATIONS:
            run(
                solver,
                args.instances,
                args.timeout_in_seconds,
                profiling=args.profiling,
                solver_path=args.solver_path,
                should_submit=args.submit,
            )
