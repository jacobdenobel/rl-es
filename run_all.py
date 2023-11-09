import time
import subprocess

from main import STRATEGIES, ENVS

SEEDS = range(1, 31)

for env in ["BipedalWalker-v3"]:
    for strat in ["egs", "ars-v1"]:
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
                "--eval_total_timesteps",
                "--sigma0",
                "0.5",
            ], start_new_session=True)
            time.sleep(1)






