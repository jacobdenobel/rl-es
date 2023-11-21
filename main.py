import os
import time
import argparse
import json
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from algorithms import MAES, DR1, ARS, CSA, DR2, EGS, CMA_EGS, ARS_OPTIMAL_PARAMETERS, CSA_EGS
from objective import Objective, ENVIRONMENTS

DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")
STRATEGIES = ("maes", "dr1", "csa", "dr2",  "ars", "ars-v2", "egs", "cma-egs", "csa-egs")

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
        default=None,
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
    parser.add_argument("--regularized", action="store_true")
    parser.add_argument("--seed_train_envs", action="store_true")
    parser.add_argument("--scale_by_std", action="store_true")
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
    parser.add_argument("--eval_total_timesteps", action="store_true")

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
    
    strategy_name = args.strategy
    if not args.strategy.startswith("ars"):
        if args.uncertainty_handled:
            strategy_name =  f'UH-{strategy_name}'
        if args.mirrored:
            strategy_name =  f'{strategy_name}-mirrored'
        if args.regularized:
            strategy_name =  f'{strategy_name}-reg'
        if args.normalized:
            strategy_name =  f'{strategy_name}-norm'
        if args.sigma0 is None:
            strategy_name =  f'{strategy_name}-sigma-default'
            n = env_setting.action_size * env_setting.state_size
            args.sigma0 = min(max(1 / n, 0.005), .1)
            print("using sigma0", args.sigma0)
        else:
            strategy_name =  f'{strategy_name}-sigma-{args.sigma0:.2e}'

        if args.scale_by_std:
            strategy_name =  f'{strategy_name}-std'

    data_folder = f"{DATA}/{args.env_name}/{strategy_name}/{t}"
    if args.play is None:
        os.makedirs(data_folder, exist_ok=True)

    if args.strategy == "ars-v2":
        args.normalized = True
        args.regularize = False
    elif args.strategy == "ars":
        args.normalized = False
        args.regularize = False

    obj = Objective(
        env_setting,
        args.n_episodes,
        args.n_timesteps,
        args.n_hidden,
        args.n_layers,
        normalized=args.normalized,
        bias=args.with_bias,
        eval_total_timesteps=args.eval_total_timesteps,
        n_test_episodes=args.n_test_episodes,
        store_video=args.store_videos,
        data_folder=data_folder,
        regularized=args.regularized,
        seed_train_envs=args.seed if args.seed_train_envs else None
    )
    if args.play is None:
        obj.open()
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
                mirrored=args.mirrored,
                scale_by_std=args.scale_by_std
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

        elif args.strategy == "ars" or args.strategy == "ars-v2":
            if args.ars_optimal and (params:=ARS_OPTIMAL_PARAMETERS.get(args.env_name)):
                args.alpha = params.alpha
                args.sigma0 = params.sigma
                args.lamb = params.lambda0
                args.mu = params.mu

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
        elif args.strategy == "csa-egs":
            optimizer = CSA_EGS(
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
        weights = np.load(f"{args.play}.npy")
        
        obj.normalizer.mean = np.load(f"{args.play}-norm-mean.npy")
        obj.normalizer.std = np.load(f"{args.play}-norm-std.npy")
        obj.store_video = True
        obj.n_test_episodes = 1
        obj.data_folder = os.path.dirname(data_folder)
        obj.test(weights, render_mode="rgb_array_list", name="test")
    
    # best_test = obj.play(best, data_folder, "best", plot)
    # print("Test with best x (median max):", best_test)
    # mean_test = obj.play(mean, data_folder, "mean", plot)
    # print("Test with mean x (median max):", mean_test)

    time.sleep(1)
