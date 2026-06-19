from functools import reduce
from pathlib import Path
from typing import Tuple

CONFIGURATIONS = [
    "geas",
    "geas-free",
    "ortools",
    "ortools-free",
    "chuffed",
    "chuffed-free",
    "pumpkin-no-faithfulness",
    "pumpkin-no-faithfulness-free",
    "pumpkin-faithfulness",
    "pumpkin-faithfulness-free",
    "pumpkin-faithfulness-nominimization-free",
    "pumpkin-faithfulness-decisionlearning-free",
    "pumpkin-main",
    "pumpkin-main-updated",
    "pumpkin-dev",
    "pumpkin-dev-updated",
    "pumpkin-dev-updated-100MB",
    "pumpkin-fix",
    "pumpkin-v0",
    "pumpkin-v1",
    "pumpkin-v2",
    "pumpkin-v3",
]

EXPERIMENTS_DIR_PREFIX = Path("./experiments").resolve()
INSTANCES_DIR_PREFIX = Path("./processed_new").resolve()
IMAGES_DIR_PREFIX = Path("./images").resolve()

CONFIG_TO_SOLVER = {
    "geas": "geas",
    "chuffed": "chuffed",
    "ortools": "ortools",
    "pumpkin-no-faithfulness": "pumpkin",
    "pumpkin-faithfulness": "pumpkin",
    "geas-free": "geas",
    "chuffed-free": "chuffed",
    "ortools-free": "ortools",
    "pumpkin-no-faithfulness-free": "pumpkin",
    "pumpkin-faithfulness-free": "pumpkin",
    "pumpkin-faithfulness-nominimization-free": "pumpkin",
    "pumpkin-faithfulness-decisionlearning-free": "pumpkin",
    "pumpkin-main": "pumpkin",
    "pumpkin-main-updated": "pumpkin",
    "pumpkin-dev": "pumpkin",
    "pumpkin-dev-updated": "pumpkin",
    "pumpkin-dev-updated-100MB": "pumpkin",
    "pumpkin-fix": "pumpkin",
    "pumpkin-v0": "pumpkin-v0",
    "pumpkin-v1": "pumpkin-v1",
    "pumpkin-v2": "pumpkin-v2", 
    "pumpkin-v3": "pumpkin-v3",
}

SOLVER_DIRS = {
    "pumpkin-main": Path("./solvers/Pumpkin").resolve(),
    "pumpkin-main-updated": Path("./solvers/PumpkinUpdated").resolve(),
    "pumpkin-dev": Path("./solvers/PumpkinDev").resolve(),
    "pumpkin-dev-updated": Path("./solvers/PumpkinDevUpdated").resolve(),
    "pumpkin-fix": Path("./solvers/PumpkinFix").resolve(),
    "pumpkin-v0": Path("./solvers/pumpkin-v0").resolve(),
    "pumpkin-v1": Path("./solvers/pumpkin-v1").resolve(),
    "pumpkin-v2": Path("./solvers/pumpkin-v2").resolve(),
    "pumpkin-v3": Path("./solvers/pumpkin-v3").resolve(),
}

SOLVER_TO_FLAGS = {}

SOLVER_CONFIGS = {
    "geas": Path("./solvers/geas/fzn/geas.msc").resolve(),
    "ortools": Path("./solvers/or-tools-9.12/build/cp-sat.msc").resolve(),
    "chuffed": Path("./solvers/chuffed-0.13.2/build/chuffed.msc").resolve(),
    "pumpkin-no-faithfulness": Path("./solvers/pumpkin-no-faithfulness/minizinc/pumpkin.msc").resolve(),
    "pumpkin-faithfulness": Path("./solvers/pumpkin-domainfaithfulness/minizinc/pumpkin.msc").resolve(),
    "geas-free": Path("./solvers/geas/fzn/geas.msc").resolve(),
    "ortools-free": Path("./solvers/or-tools-9.12/build/cp-sat.msc").resolve(),
    "chuffed-free": Path("./solvers/chuffed-0.13.2/build/chuffed.msc").resolve(),
    "pumpkin-no-faithfulness-free": Path("./solvers/pumpkin-no-faithfulness/minizinc/pumpkin.msc").resolve(),
    "pumpkin-faithfulness-free": Path("./solvers/pumpkin-domainfaithfulness/minizinc/pumpkin.msc").resolve(),
    "pumpkin-faithfulness-nominimization-free": (
        Path("./solvers/pumpkin-domainfaithfulness/minizinc/pumpkin.msc").resolve()
    ),
    "pumpkin-faithfulness-decisionlearning-free": (
        Path("./solvers/pumpkin-domainfaithfulness/minizinc/pumpkin.msc").resolve()
    ),
    "pumpkin-main": Path("./solvers/Pumpkin/minizinc/pumpkin.msc").resolve(),
    "pumpkin-main-updated": Path("./solvers/PumpkinUpdated/minizinc/pumpkin.msc").resolve(),
    "pumpkin-dev": Path("./solvers/PumpkinDev/minizinc/pumpkin.msc").resolve(),
    "pumpkin-dev-updated": Path("./solvers/PumpkinDevUpdated/minizinc/pumpkin.msc").resolve(),
    "pumpkin-dev-updated-100MB": Path("./solvers/PumpkinDevUpdated/minizinc/pumpkin.msc").resolve(),
    "pumpkin-fix": Path("./solvers/PumpkinFix/minizinc/pumpkin.msc").resolve(),
    "pumpkin-v0": Path("/scratch/mcarrio/evaluation/solvers/pumpkin-v0/minizinc/pumpkin.msc"),
    "pumpkin-v1": Path("/scratch/mcarrio/evaluation/solvers/pumpkin-v1/minizinc/pumpkin.msc"),
    "pumpkin-v2": Path("/scratch/mcarrio/evaluation/solvers/pumpkin-v2/minizinc/pumpkin.msc"),
    "pumpkin-v3": Path("/scratch/mcarrio/evaluation/solvers/pumpkin-v3/minizinc/pumpkin.msc"),
}

INSTANCES = {
    "geas": INSTANCES_DIR_PREFIX / "geas",
    "ortools": INSTANCES_DIR_PREFIX / "ortools",
    "chuffed": INSTANCES_DIR_PREFIX / "chuffed",
    "pumpkin-no-faithfulness": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-faithfulness": INSTANCES_DIR_PREFIX / "pumpkin",
    "geas-free": INSTANCES_DIR_PREFIX / "geas",
    "ortools-free": INSTANCES_DIR_PREFIX / "ortools",
    "chuffed-free": INSTANCES_DIR_PREFIX / "chuffed",
    "pumpkin-no-faithfulness-free": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-faithfulness-free": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-faithfulness-nominimization-free": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-faithfulness-decisionlearning-free": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-main": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-main-updated": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-dev": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-dev-updated": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-dev-updated-100MB": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-fix": INSTANCES_DIR_PREFIX / "pumpkin",
    "pumpkin-v0": INSTANCES_DIR_PREFIX / "pumpkin-v0",
    "pumpkin-v1": INSTANCES_DIR_PREFIX / "pumpkin-v1",
    "pumpkin-v2": INSTANCES_DIR_PREFIX / "pumpkin-v2",
    "pumpkin-v3": INSTANCES_DIR_PREFIX / "pumpkin-v3"
}


assert SOLVER_CONFIGS.keys() == INSTANCES.keys()
assert INSTANCES.keys() == set(CONFIGURATIONS)
assert set(CONFIGURATIONS) == CONFIG_TO_SOLVER.keys(), "Every configuration must map to a solver."


def split(string, sep, n) -> Tuple[str, str]:
    """Split `string´ at the `n`th occurrence of `sep`"""
    pos = reduce(lambda x, _: string.index(sep, x + 1), range(n + 1), -1)
    return string[:pos], string[pos + len(sep) :]
