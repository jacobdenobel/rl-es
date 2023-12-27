import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from objective import Objective, ENVIRONMENTS


def load_data(env_folder):
    stats_list = []
    for strat in os.listdir(env_folder):
        path = os.path.join(env_folder, strat)
        if not os.path.isdir(path): continue
        for i, (run) in enumerate(os.listdir(path)):
            path = os.path.join(env_folder, strat, run)
            if not os.path.isdir(path): continue
            stats = pd.read_csv(os.path.join(path, "stats.csv"), skipinitialspace=True)
            stats['run'] = i
            stats['folder'] = run
            stats['strat'] = strat
            sig, lamb = strat.split("sigma-")[1].split("-lambda-")
            stats['sigma0'] = float(sig)
            stats['lambda'] = int(lamb)
            stats_list.append(stats)
        
    return pd.concat(stats_list, ignore_index=True)

def get_objective(env_name, store_videos=False, data_folder=''):
    env_setting = ENVIRONMENTS[env_name]
    return Objective(
        env_setting,
        1,
        None,
        8,
        1,
        normalized=True,
        bias=False,
        eval_total_timesteps=False,
        n_test_episodes=5,
        store_video=store_videos,
        data_folder=data_folder,
        regularized=False,
        seed_train_envs=None,
    )  


def get_loc(env_name, record):
    root = os.path.join("./data", env_name, record.strat, record.folder, "policies", f"t-{record.generation}")
    best = f"{root}-best"
    current = f"{root}-mean"
    assert os.path.isfile(f"{best}.npy") and os.path.isfile(f"{current}.npy")
    return best, current

def get_auc(data):
    df_auc = pd.DataFrame(columns=["method", 'sigma0', 'lambda', 'auc'])
    for _, group in data.groupby("method"):
        time = np.sort(np.unique(group.n_evals))
        setting_test = []
        setting_parm = []
        for _, setting, in group.groupby("strat"):
            y = setting.groupby("n_evals").expected_test.median()
            setting_test.append(np.interp(time, y.index.values, y.values))
            setting_parm.append((setting.method.iloc[0], setting.sigma0.iloc[0], setting['lambda'].iloc[0]))
        
        setting_test = np.array(setting_test)
        setting_test += -setting_test.min()
        setting_auc = np.array([np.trapz(si, time) for si in setting_test])
        df_auc = pd.concat([df_auc, pd.DataFrame(np.c_[setting_parm, setting_auc], columns=df_auc.columns)])

    df_auc = df_auc.astype({'sigma0': float, 'lambda': int, "auc": float})
    top_performers = df_auc.groupby("method")['auc'].max()
    top_performers = df_auc[df_auc.auc.isin(top_performers)].drop_duplicates(subset='method', keep='first')
    assert len(top_performers) == 3
    return df_auc, top_performers

ENVS = [
    # "Swimmer-v4",
    # "HalfCheetah-v4",
    # "Hopper-v4",
    # "Walker2d-v4",
    # "Ant-v4",
    # "Humanoid-v4",
    # "LunarLander-v2"
    # "CartPole-v1",
    # "Acrobot-v1",
    # "Pendulum-v1",
    "BipedalWalker-v3"
]

if __name__ == "__main__":

    for env_name in ENVS:
        obj = get_objective(env_name)
        data = load_data(f"data/{env_name}")
        data = data[data.generation % 5 == 0].reset_index(drop=True)
        data['train'] = data[['best', 'current']].max(axis=1)
        data['method'] = data.strat.str.split("-norm-").str[0]
        data['test'] = -np.inf
        data['expected_test'] = data[['best_median', 'current_median']].max(axis=1)
        
        df_auc, top_performers = get_auc(data)

        print(env_name)
        print(top_performers)
        cols = ["method","sigma0", "lambda"]
        index = pd.MultiIndex.from_frame(data[cols])
        data = data[index.isin(top_performers[cols].values.tolist())]

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            best_loc, current_loc = get_loc(env_name, row)
            if row.best_median > row.current_median:
                loc = best_loc 
                expected = row.best_median
            else:
                loc = current_loc
                expected = row.current_median
            # breakpoint()  
            data.at[idx, "test"] = expected
            # if row.method.startswith("sep"):
            # else:
            #     data.at[idx, "test"] = np.median(obj.play_check(loc, n_reps=5))

        data = data[['method', 'sigma0', 'lambda', 'run', 'generation', 'n_train_episodes', 'n_train_timesteps', 'test', 'train', 'expected_test']]
        data.to_pickle(f"data/{env_name}/data.pkl")
