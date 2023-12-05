import numpy as np

from exploration_on_policy.utils.normalizer import TorchRunningMeanStd
from exploration_on_policy.utils.utils import obvs_preprocess


def collect_inital_stats(envs,dim_x,dim_y, args, i_agent):
    print("Start to initialize observation normalization parameter.....")
    envs.reset()
    obs_stats = TorchRunningMeanStd((dim_x, dim_y))
    i_reward_stats = TorchRunningMeanStd((args.num_envs,envs.action_space.n))
    for step in range(1000):
        acs = np.random.randint(0, envs.action_space.n, size=(args.num_envs,))
        s, r, d, _ = envs.step(acs)
        t_obs = obvs_preprocess(s)
        obs_stats.update_single(t_obs)
        rew = i_agent.pseudo_ucb_nets(t_obs)
        i_reward_stats.update_single(rew)
    print("End to initialize...")
    return obs_stats, i_reward_stats
