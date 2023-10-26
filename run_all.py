import subprocess

from main import STRATEGIES, ENVS, BUDGETS

SEEDS = range(10)

for env in ["LunarLander-v2"]:
    for strat in STRATEGIES:
        for seed in SEEDS:
            subprocess.run([
                "python",
                "main.py",
                "--env_name",
                env,
                "--strategy",
                strat,
                "--seed",
                str(seed),
                "--eval_total_timesteps",
            ])






