import os
import time
import subprocess
from itertools import product

from objective import ENVIRONMENTS
from algorithms import init_lambda

ENVS = [
    # "Swimmer-v4",
    # "HalfCheetah-v4",
    # "Hopper-v4",
    # "Walker2d-v4",
    # "Ant-v4",
    "Humanoid-v4",
    # "LunarLander-v2", 
    # "BipedalWalker-v3",
  
]

SEEDS = [x * 12 for x in range(0, 10)]
STRATEGIES = [
    "csa",
    # "cma-es",
    # "sep-cma-es",
]

SIGMA = [
    0.1,
    0.05,
    0.01,
]

LAMBDA = [
    # 4, 
    # "default", 
    # "n/2"
    128, 
    256
]
 
open_files = []
for env in ENVS:
    meta = ENVIRONMENTS.get(env)
    n = meta.action_size * meta.state_size
    parameters = list(
        product(STRATEGIES, SIGMA, [init_lambda(n, l) for l in LAMBDA], SEEDS)
    )
    n_proc = len(parameters)
    print(f"Spawning {n_proc} processess 2s to stop")
    time.sleep(2)

    for i, (strat, sigma, lamb, seed) in enumerate(parameters):
        if i % 10 == 0:
            time.sleep(50)
            load1, load5, load15 = os.getloadavg() 
            while load1 > 150:
                print(" load too high, sleeping 10 seconds...", end="\r", flush=True)
                time.sleep(10)
                load1, load5, load15 = os.getloadavg() 
                print(f" proc {i}/{n_proc}, load (1, 5, 15): ({load1, load5, load15})", end=" ")

        stdout = open(f"run/{i}", "w+") 
        command = [
                "python",
                "main.py",
                "--env_name",
                env,
                "--strategy",
                strat,
                "--seed",
                str(seed),
                "--normalized",
                "--sigma0",
                str(sigma),
                "--lamb",
                str(lamb),
        ]
        subprocess.Popen(command,
            start_new_session=True,
            env=dict(os.environ, MUJOCO_GL="egl"),
            stdout=stdout,
            stderr=stdout,
        )
        open_files.append(stdout)
        print(f"proc {i}/{n_proc}: " + " ".join(command))

for f in open_files:
    f.close()