import time
import subprocess

from main import STRATEGIES, ENVS

SEEDS = range(10, 20)

for env in ["LunarLander-v2"]:
    for strat in STRATEGIES[:-1]:
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
                "0.001"                
            ], start_new_session=True)
            time.sleep(1)






