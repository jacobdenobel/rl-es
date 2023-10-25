import os
import time
import argparse
import json

import numpy as np
import gymnasium as gym
from algorithms import MAES, DR1, ARSV1, CSA, DR2
from objective import Objective

DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")

ENVS = (
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v2",
    "BipedalWalker-v3",
)

BUDGETS = (
    500,       # Cartpole
    1000,       # Acrobot
    1000,       # MountainCar
    10_000,     # LunarLander
    20_000      # Walker
)


STRATEGIES = ("maes", "dr1", "ars-v1", "csa", "dr2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget", default=None, help="Number of fitness evaluations", type=int
    )
    parser.add_argument(
        "--seed", default=42, help="Set seed for reproducibility", type=int
    )
    parser.add_argument(
        "--n_episodes",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_test_episodes",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--test_every_nth_iteration",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--n_timesteps",
        help="Number of timesteps for each training episode",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mu",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lamb",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n_hidden",
        help="Number of hidden units in layer of the neural network (only used if n_layers > 2)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_layers", help="Number of layers in the controller", type=int, default=1
    )
    parser.add_argument("--with_bias", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--uncertainty_handled", action="store_true")
    parser.add_argument(
        "--initialization",
        type=str,
        choices=("zero", "uniform", "gauss"),
        default="zero",
    )
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="csa")

    parser.add_argument("--env_name", type=str, default="LunarLander-v2", choices=ENVS)
    parser.add_argument("--eval_total_timesteps", action="store_false")
    parser.add_argument(
        "--play",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()

    t = time.time()

    if not os.path.isdir(DATA):
        os.makedirs(DATA)

    spec = gym.make(args.env_name).spec
    if args.n_timesteps is None:
        args.n_timesteps = spec.max_episode_steps

    if args.budget is None:
        args.budget = BUDGETS[ENVS.index(args.env_name)]

    print(args)
    print(spec)

    np.random.seed(args.seed)
    obj = Objective(
        args.n_episodes,
        args.n_timesteps,
        args.n_hidden,
        args.n_layers,
        env_name=args.env_name,
        normalized=args.normalized,
        no_bias=not args.with_bias,
        single_episode_per_eval=args.eval_total_timesteps,
        n_test_episodes=args.n_test_episodes,
    )
    plot = True
    data_folder = f"{DATA}/{args.env_name}/{args.strategy}/{t}"
    if args.play is None:
        os.makedirs(data_folder)
        if args.strategy == "maes":
            optimizer = MAES(
                obj.n,
                args.budget,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
            )
        elif args.strategy == "dr1":
            optimizer = DR1(
                obj.n,
                args.budget,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
            )
        elif args.strategy == "dr2":
            optimizer = DR2(
                obj.n,
                args.budget,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
            )
        elif args.strategy == "csa":
            optimizer = CSA(
                obj.n,
                args.budget,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
            )

        elif args.strategy == "ars-v1":
            optimizer = ARSV1(
                obj.n,
                args.budget,
                data_folder=data_folder,
                test_gen=args.test_every_nth_iteration,
            )
        else:
            raise ValueError()

        args.sigma0 = optimizer.sigma0
        args.mu = int(optimizer.mu)
        args.lambda_ = int(optimizer.lambda_)
        args.n = int(obj.n)

        with open(os.path.join(data_folder, "settings.json"), "w+") as f:
            json.dump(vars(args), f)

        best, mean = optimizer(obj)
        np.save(f"{data_folder}/best.npy", best.x)
        np.save(f"{data_folder}/mean.npy", mean.x)
        best, mean = best.x, mean.x
    else:
        data_folder = args.play
        best = np.load(os.path.join(args.play, "best.npy"))
        mean = np.load(os.path.join(args.play, "mean.npy"))
        plot = False
    
    # best_test = obj.play(best, data_folder, "best", plot)
    # print("Test with best x (median max):", best_test)
    # mean_test = obj.play(mean, data_folder, "mean", plot)
    # print("Test with mean x (median max):", mean_test)

    time.sleep(1)