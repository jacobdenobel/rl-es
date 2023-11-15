import os
import time
import subprocess

from main import STRATEGIES

SEEDS = range(20, 26)

for env in ['Hopper-v4']:
    for strat in ("maes", "ars", "csa", "cma-egs"):
        for seed in SEEDS:
            subprocess.Popen([
                "python",
                "main.py",
                "--env_name",
                env,
                "--strategy",
                strat,
                "--seed",
                str(seed),
                "--sigma0",
                "0.1",
                "--normalized",
                "--ars_optimal"
            ], start_new_session=True, env=dict(os.environ, MUJOCO_GL="egl"))
            time.sleep(1)






