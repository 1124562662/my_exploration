import argparse
import math
import os
import random
import statistics
import time
from distutils.util import strtobool
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import numpy as np

from exploration_on_policy.Buffers.RND_diversity_aware_buffer import RNDReplayBuffer
from exploration_on_policy.Buffers.ReplayBuffer_eopg import ReplayBuffer

from exploration_on_policy.Extrinsic_agent import ExtrinsicQAgent
from exploration_on_policy.All_intrinsic_agents.agent_rnd_offpolicy_buffer import IntrinsicAgent
from exploration_on_policy.utils.Initial_stats_rollout import collect_inital_stats
from exploration_on_policy.utils.normalizer import TorchRunningMeanStd
from exploration_on_policy.utils.utils import obvs_preprocess

""" use rnd as (s,a) psudo count and novelD """


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=2,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
                        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ALE/MontezumaRevenge-v5",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--i_learning_rate", type=float, default=1e-3,
                        help="the learning rate of the optimizer")
    parser.add_argument("--e_learning_rate", type=float, default=1e-3,
                        help="the learning rate of the optimizer")
    parser.add_argument("--pseudo_ucb_optimizer_lr", type=float, default=1e-8,
                        help="learning rate of pseudo_ucb_optimizer")
    parser.add_argument("--num-envs", type=int, default=2,
                        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=100,
                        help="the replay memory buffer size")

    parser.add_argument("--extrinsic_update_target_tau", type=float, default=0.1,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--target-network-frequency", type=int, default=5,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=160,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.3,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1,
                        help="timestep to start learning")

    parser.add_argument("--e_target_network_frequency", type=int, default=6,
                        help="the frequency of training")
    parser.add_argument("--train_extrinsic_frequency", type=int, default=2,
                        help="the frequency of training")
    parser.add_argument("--E_action_gumbel_max_tau", type=float, default=0.99,
                        help="used for gumbel softmax of extrinsic agent")
    parser.add_argument("--I_action_gumbel_max_tau", type=float, default=0.99,
                        help="used for gumbel softmax of intrinsic agent")
    parser.add_argument("--I_ubc_gumbel_max_tau", type=float, default=0.99,
                        help="used for gumbel softmax of intrinsic agent")
    parser.add_argument("--pseudo_ucb_coef", type=float, default=0.5,
                        help="used for pseudo_ucb_coef of intrinsic agent")
    parser.add_argument("--embed_dim", type=int, default=70,
                        help="embedding size of intrinsic Q(s,a)")

    parser.add_argument("--log_interval", type=int, default=1,
                        help="log_interval")
    parser.add_argument("--train_intrinsic_epoches", type=int, default=4,
                        help=" train_intrinsic_epoches for the intrinsic agent")
    parser.add_argument("--train_extrinsic_epoches", type=int, default=4,
                        help=" train_extrinsic_epoches for the extrinsic agent")
    parser.add_argument("--intrinsic_max_loss", type=float, default=0.1,
                        help=" intrinsic_max_loss")
    parser.add_argument("--clip_intrinsic_reward_min", type=float, default=0.0,
                        help=" if the one step intrinsic reward is less than this threshold, the reward is clipped to 0")

    parser.add_argument("--render-human", type=bool, default=True,
                        help=" ")
    parser.add_argument("--num-steps", type=int, default=240,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--rnd-train-freq", type=int, default=12,
                        help=" ")

    parser.add_argument("--gamma", type=float, default=0.95,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--int_gamma", type=float, default=0.99,
                        help="Intrinsic reward discount rate")

    parser.add_argument("--net_num", type=int, default=3,
                        help=" ")
    parser.add_argument("--train_net_num", type=int, default=3,
                        help="")
    parser.add_argument("--use_e_agent", type=bool, default=False,
                        help=" ")
    parser.add_argument("--rnd_train_num", type=int, default=40,
                        help=" ")
    parser.add_argument("--rnd_update_epochs", type=int, default=1,
                        help=" ")
    parser.add_argument("--novelD-alpha", type=float, default=0.5,
                        help="novelD-alpha if positive, do not use novelD if zero")

    parser.add_argument("--extrinsic_rewards_for_i_agent", type=float, default=1,
                        help=" ")
    # parser.add_argument("--use-contextual-bandit-UBC", type=bool, default=False,
    #                     help=" ")
    parser.add_argument("--use_only_UBC_exploration_threshold", type=float, default=0.3,
                        help=" ")
    # buffer related
    parser.add_argument("--rnd_buffer_size", type=int, default=3,
                        help=" ")
    parser.add_argument("--ema_beta", type=float, default=0.3,
                        help=" ")
    parser.add_argument("--initial_traj_len_times", type=int, default=2,
                        help=" ")
    parser.add_argument("--buffer_encoder_emb_dim", type=int, default=100,
                        help=" ")
    parser.add_argument("--encoder_learning_rate", type=float, default=0.001,
                        help=" ")
    parser.add_argument("--initial_encoder_train_epoches", type=int, default=10,
                        help=" ")
    parser.add_argument("--train_with_buffer_interval", type=int, default=3,
                        help=" ")
    parser.add_argument("--rnd_buffer_train_off_policy_times", type=int, default=20,
                        help=" ")
    parser.add_argument("--rnd_buffer_train_off_policy_epoches", type=int, default=5,
                        help=" ")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.buffer_sample_bsize = int(math.floor(args.num_envs / args.initial_traj_len_times))
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


def make_env(env_id, rank, seed=0):
    from stable_baselines3.common.utils import set_random_seed
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            n_envs=args.num_envs
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    from stable_baselines3.common.env_util import make_vec_env

    envs = make_vec_env(args.env_id, n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)

    dim_x = 70
    dim_y = 53

    i_agent = IntrinsicAgent(envs=envs,
                             args=args,
                             device=device,
                             dim_x=dim_x,
                             dim_y=dim_y,
                             learning_rate=args.i_learning_rate,
                             I_action_gumbel_max_tau=args.I_action_gumbel_max_tau,
                             I_ubc_gumbel_max_tau=args.I_ubc_gumbel_max_tau,
                             pseudo_ucb_coef=args.pseudo_ucb_coef,
                             ae_sin_size=128,
                             filter_size=3,
                             hidden_units=512,
                             )

    if args.use_e_agent:
        e_agent = ExtrinsicQAgent(envs,
                                  device,
                                  args.extrinsic_update_target_tau,
                                  learning_rate=args.e_learning_rate,
                                  E_action_gumbel_max_tau=args.E_action_gumbel_max_tau)

        # rb = ReplayBuffer(
        #     args.buffer_size,
        #     envs.observation_space,
        #     envs.action_space,
        #     device,
        #     dim_x=dim_x,
        #     dim_y=dim_y,
        #     optimize_memory_usage=True,
        #     handle_timeout_termination=False,
        #     n_envs=args.num_envs,
        # )

    from collections import deque

    ep_info_buffer = deque(maxlen=args.num_steps)
    ep_success_buffer = deque(maxlen=args.num_steps)

    start_time = time.time()
    next_obs = envs.reset()
    next_obs = obvs_preprocess(next_obs, device=device)  # ,obs_stats= self.obs_stats)

    # TRY NOT TO MODIFY: start the game
    # obs, _ = envs.reset(seed=args.seed)

    total_env_step_call = 0
    global_i_step = 0
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if args.use_e_agent and (global_step % 100) / 100 > epsilon:
            pass
            # current_agent = e_agent
            # e_agent.rollout_step(args, device, dim_x, dim_y, envs, ep_info_buffer, ep_success_buffer, rb, global_step,
            #                      next_obs)
        else:
            current_agent = i_agent
            mean_i_rewards, max_i_rewards, next_obs = i_agent.rollout_step(args, device, dim_x, dim_y, envs,
                                                                           ep_info_buffer,
                                                                           ep_success_buffer, global_step, next_obs)
            global_i_step += 1
            if global_step % args.log_interval == 0:
                print("global_step", global_step, ",   total_env_step_call", total_env_step_call, ",   I_rew",
                      mean_i_rewards, ",  max i rewards", max_i_rewards)

        total_env_step_call += args.num_envs * args.num_steps

        if global_step % args.log_interval == 0:
            assert ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
            if len(ep_info_buffer) > 0:
                if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
                    print("rollout/ep_rew_mean", statistics.mean([ep_info["r"] for ep_info in ep_info_buffer]),
                          ",   ep_len_mean",
                          statistics.mean([ep_info["l"] for ep_info in ep_info_buffer]))

        # if args.save_model:
        #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        #     torch.save(q_network.state_dict(), model_path)
        #     print(f"model saved to {model_path}")
        #     from cleanrl_utils.evals.dqn_eval import evaluate
        #
        #     episodic_returns = evaluate(
        #         model_path,
        #         make_env,
        #         args.env_id,
        #         eval_episodes=10,
        #         run_name=f"{run_name}-eval",
        #         Model=QNetwork,
        #         device=device,
        #         epsilon=0.05,
        #     )
        #     for idx, episodic_return in enumerate(episodic_returns):
        #         writer.add_scalar("eval/episodic_return", episodic_return, idx)
        #
        #     if args.upload_model:
        #         from cleanrl_utils.huggingface import push_to_hub
        #
        #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #         push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
