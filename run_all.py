import time
import subprocess

from main import STRATEGIES, ENVS

SEEDS = range(10, 20)

for env in ["BipedalWalker-v3"]:
    for strat in STRATEGIES[-1:]:
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
                "0.02023",
                "--lamb",
                "32"
            ], start_new_session=True)
            time.sleep(1)






