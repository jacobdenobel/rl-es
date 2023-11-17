import os
import time
import subprocess

from main import STRATEGIES

SEEDS = range(31, 36)

for env in ['Ant-v4',]:
    for strat in ("maes", ):
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
                "0.02",
                "--normalized",
                "--ars_optimal",
                "--regularize"
            ], start_new_session=True, env=dict(os.environ, MUJOCO_GL="egl"))
            time.sleep(1)






