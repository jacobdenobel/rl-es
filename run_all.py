import os
import time
import subprocess

from main import STRATEGIES
from objective import CLASSIC_CONTROL, BOX2D, MUJOCO

SEEDS = range(0, 10)
strats = [
    # "csa", 
    # "active-cma-es", 
    # "active-sep-cma-es", 
    # "cma-es", 
    "maes",
    "sep-cma-es",
    "sep-cma-egs",
]
for env in ["Breakout-v5"]:
    for strat in strats:
        for seed in SEEDS:
            subprocess.Popen(
                [
                    "python",
                    "main.py",
                    "--env_name",
                    env,
                    "--strategy",
                    strat,
                    "--seed",
                    str(7 * seed),
                    "--normalized",
                    "--ars_optimal",
                    "--store_videos",
                ],
                start_new_session=True,
                env=dict(os.environ, MUJOCO_GL="egl"),
            )

    done = True
    while not done:
        time.sleep(10)
        out = int(
            subprocess.check_output(
                "ps -aux | grep main.py | grep -v grep | wc -l",
                shell=True,
                universal_newlines=True,
            )
        )
        done = out == 0
