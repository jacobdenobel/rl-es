import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def align(time, x, y):
    return np.interp(time, x, y)


def plot_whitebg(ax, time, y, label, color):
    ax.plot(
        time,
        y,
        color="white",
        linewidth=3,
        alpha=1,
    )

    ax.plot(
        time,
        y,
        color=color,
        label=label,
        linewidth=2,
        alpha=1,
    )


def plot_all_strats_for_env(env_folder, dt=500):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    time = None
    for strat, color in zip(os.listdir(env_folder), colors.TABLEAU_COLORS):
        means_best = []
        means_current = []
        test_means_best = []
        test_means_current = []

        for i, (run) in enumerate(os.listdir(os.path.join(env_folder, strat))):
            path = os.path.join(env_folder, strat, run)
            stats = pd.read_csv(os.path.join(path, "stats.csv"), skipinitialspace=True)
            with open(os.path.join(path, "settings.json")) as f:
                meta = json.load(f)

            if time is None:
                time = np.arange(0, meta["budget"] * meta['n_timesteps'], dt)

            means_best.append(np.interp(time, stats.n_train_timesteps, stats.best))
            means_current.append(np.interp(time, stats.n_train_timesteps, stats.current))
            test_means_best.append(np.interp(time, stats.n_train_timesteps, stats.best_test))
            test_means_current.append(np.interp(time, stats.n_train_timesteps, stats.current_test))

            ax1.plot(time, means_best[-1], color=color, alpha=0.2, zorder=-1)
            ax2.plot(time, means_current[-1], color=color, alpha=0.2, zorder=-1)
            ax3.plot(time, test_means_best[-1], color=color, alpha=0.2, zorder=-1)
            ax4.plot(time, test_means_current[-1], color=color, alpha=0.2, zorder=-1)

        plot_whitebg(ax1, time, np.median(means_best, axis=0), strat, color)
        plot_whitebg(ax2, time, np.median(means_current, axis=0), strat, color)
        plot_whitebg(ax3, time, np.median(test_means_best, axis=0), strat, color)
        plot_whitebg(ax4, time, np.median(test_means_current, axis=0), strat, color)

    ax1.set_title("best return (train)")
    ax2.set_title("mean return (train)")

    ax3.set_title("best return (test)")
    ax4.set_title("mean return (test)")
    for ax in ax1, ax2, ax3, ax4:
        ax.set_ylabel("returns")
        ax.set_xlabel("# train timesteps")
        ax.legend()
        ax.grid()
    plt.suptitle((meta["env_name"]))
    plt.savefig(meta['env_name'] + ".png")
    #plt.show()


def plot_x_final(env_folder):
    for strat, color in zip(os.listdir(env_folder), colors.TABLEAU_COLORS):
        means = []
        for i, (run) in enumerate(os.listdir(os.path.join(env_folder, strat))):
            path = os.path.join(env_folder, strat, run)
            stats = pd.read_csv(os.path.join(path, "stats.csv"), skipinitialspace=True)
            with open(os.path.join(path, "settings.json")) as f:
                meta = json.load(f)
            
            policy_folder = os.path.join(path, "policies")
            if not os.path.isdir(policy_folder):
                continue
            
            policy, *_ = sorted(
                [x for x in os.listdir(policy_folder) if "mean" in x],
                key=lambda x: -int(x.split("-")[1])
            )
            policy_vector = np.load(os.path.join(policy_folder, policy))
            means.append(policy_vector.ravel())

        plt.plot(np.mean(means, axis=0), label=strat)

    plt.legend()
    plt.grid()
    plt.savefig(meta['env_name'] + "coordinates.png")
             




if __name__ == "__main__":
    envs = os.listdir("data")
    for env in envs:
        if env.startswith("old"): continue
        env_dir = os.path.join("data", env)
#        plot_x_final(env_dir)
        plot_all_strats_for_env(env_dir)
