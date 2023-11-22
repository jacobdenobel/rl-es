import os
import time
import subprocess

from main import STRATEGIES
from objective import CLASSIC_CONTROL, BOX2D, MUJOCO

SEEDS = range(0, 10)

for env in ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4", "Ant-v4"]:
    time.sleep(2 * 60 * 60)
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
                str(7 * seed),
#                "--sigma0",
#                "0.1",
                "--normalized",
                "--ars_optimal",
            ], start_new_session=True, env=dict(os.environ, MUJOCO_GL="egl"))

    done = True
    while not done:
        time.sleep(10)
        out = int(subprocess.check_output(
            "ps -aux | grep main.py | grep -v grep | wc -l", 
            shell=True, 
            universal_newlines=True
        ))
        done = out == 0      
        

