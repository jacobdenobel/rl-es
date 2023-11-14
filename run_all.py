import os
import time
import subprocess

from main import STRATEGIES, ENVS

SEEDS = range(1, 6)

for env in ['HalfCheetah-v4']:
    for strat in STRATEGIES:
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
                "0.5",
                "--initialization",
                "zero",
                "--normalized"
            ], start_new_session=True, env=dict(os.environ, MUJOCO_GL="egl"))
            time.sleep(1)






