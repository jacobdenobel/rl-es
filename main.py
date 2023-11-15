import os
import time
import argparse
import json
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from algorithms import MAES, DR1, ARS, CSA, DR2, EGS, CMA_EGS, ARS_OPTIMAL_PARAMETERS
from objective import Objective, ENVIRONMENTS

DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")
STRATEGIES = ("maes", "dr1", "csa", "dr2",  "ars", "egs", "cma-egs")

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
        default=5,
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
        "--sigma0",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.02,
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
    parser.add_argument("--mirrored", action="store_true")
    parser.add_argument("--uncertainty_handled", action="store_true")
    parser.add_argument("--store_videos", action="store_true")
    parser.add_argument("--ars_optimal", action="store_true")
    parser.add_argument(
        "--initialization",
        type=str,
        choices=("lhs", "zero", "uniform", "gauss"),
        default="zero",
    )

    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="csa")
    parser.add_argument(
        "--env_name", type=str, default="LunarLander-v2", 
        choices=ENVIRONMENTS.keys()
    )
    parser.add_argument("--dont_eval_total_timesteps", action="store_true")

    parser.add_argument(
        "--play",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--revaluate_best_after",
        type=int,
        default=None,
    )
    
    args = parser.parse_args()
  
    t = time.time()
    env_setting = ENVIRONMENTS[args.env_name]

    if not os.path.isdir(DATA):
        os.makedirs(DATA)
   
    if args.n_timesteps is None:
        args.n_timesteps = env_setting.max_episode_steps

    if args.budget is None:
        args.budget = env_setting.budget

    print(args)
    print(env_setting)

    np.random.seed(args.seed)
    plot = True
    uh = ''
    if args.uncertainty_handled:
        uh = 'UH-'

    mirrored = ''
    if args.mirrored and args.strategy != "ars":
        mirrored = "-mirrored"

    data_folder = f"{DATA}/{args.env_name}/{uh}{args.strategy}{mirrored}/{t}"
    os.makedirs(data_folder, exist_ok=True)
    
    obj = Objective(
        env_setting,
        args.n_episodes,
        args.n_timesteps,
        args.n_hidden,
        args.n_layers,
        normalized=args.normalized,
        no_bias=not args.with_bias,
        eval_total_timesteps=not args.dont_eval_total_timesteps,
        n_test_episodes=args.n_test_episodes,
        store_video=args.store_videos,
        data_folder=data_folder
    )
    if args.play is None:
        if args.strategy == "maes":
            optimizer = MAES(
                obj.n,
                args.budget,
                mu=args.mu, 
                lambda_=args.lamb,
                sigma0=args.sigma0,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
                mirrored=args.mirrored
            )
        elif args.strategy == "dr1":
            optimizer = DR1(
                obj.n,
                args.budget,
                sigma0=args.sigma0,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
                mirrored=args.mirrored
            )
        elif args.strategy == "dr2":
            optimizer = DR2(
                obj.n,
                args.budget,
                sigma0=args.sigma0,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
                mirrored=args.mirrored
            )
        elif args.strategy == "csa":
            optimizer = CSA(
                obj.n,
                args.budget,
                sigma0=args.sigma0,
                mu=args.mu, 
                lambda_=args.lamb,
                data_folder=data_folder,
                initialization=args.initialization,
                uncertainty_handling=args.uncertainty_handled,
                test_gen=args.test_every_nth_iteration,
                mirrored=args.mirrored
            )

        elif args.strategy == "ars":
            if args.ars_optimal and (params:=ARS_OPTIMAL_PARAMETERS.get(args.env_name)):
                args.alpha = params.alpha
                args.sigma0 = params.sigma
                args.lamb = params.lambda0

            optimizer = ARS(
                obj.n,
                args.budget,
                sigma0=args.sigma0,
                alpha=args.alpha, 
                data_folder=data_folder,
                test_gen=args.test_every_nth_iteration,
                mu=args.mu, 
                lambda_=args.lamb,
                initialization=args.initialization,
        )
        elif args.strategy == "egs":
            optimizer = EGS(
                obj.n,
                args.budget,
                data_folder=data_folder,
                test_gen=args.test_every_nth_iteration,
                sigma0=args.sigma0,
                lambda_=args.lamb,
                mu=args.mu, 
                initialization=args.initialization,
                # kappa=None
            )
        elif args.strategy == "cma-egs":
            optimizer = CMA_EGS(
                obj.n,
                args.budget,
                data_folder=data_folder,
                test_gen=args.test_every_nth_iteration,
                sigma0=args.sigma0,
                lambda_=args.lamb,
                mu=args.mu, 
                initialization=args.initialization,
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
