import time
import subprocess

from main import STRATEGIES, ENVS

SEEDS = range(1, 31)

for env in ENVS:
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
                "--eval_total_timesteps",
                "--sigma0",
                "0.05",
                "--initialization",
                "zero"
            ], start_new_session=True)
            time.sleep(1)






