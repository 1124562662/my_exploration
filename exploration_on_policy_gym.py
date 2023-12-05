import argparse
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
from ReplayBuffer_eopg import ReplayBuffer
from Extrinsic_agent import ExtrinsicQAgent
from Intrinsic_agent import IntrinsicAgent

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
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

    parser.add_argument("--target-network-frequency", type=int, default=500,
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

    parser.add_argument("--e_target_network_frequency", type=int, default=600,
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

    parser.add_argument("--num-steps", type=int, default=200,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.999,
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

    parser.add_argument("--novelD-alpha", type=float, default=0.5,
                        help="novelD-alpha if positive, do not use novelD if zero")
    parser.add_argument("--within_eposide_coef", type=float, default=0.5,
                        help="value to be determined")
    parser.add_argument("--byol_predict_horizon", type=int, default=20,
                        help=" ")
    parser.add_argument("--extrinsic_rewards_for_i_agent", type=float, default=1,
                        help=" ")
    parser.add_argument("--byol_train_epochs", type=int, default=10,
                        help=" ")
    parser.add_argument("--byol_update_embedding_to_target_freq", type=int, default=5,
                        help=" ")
    parser.add_argument("--byol_open_loop_gru_max_horizon", type=int, default=40,
                        help=" ")
    parser.add_argument("--use-contextual-bandit-UBC", type=bool, default=False,
                        help=" ") # TODO

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    from stable_baselines3.common.env_util import make_vec_env

    envs = make_vec_env(args.env_id, n_envs=args.num_envs, vec_env_cls=SubprocVecEnv)

    i_agent = IntrinsicAgent(envs=envs,
                             args=args,
                             device =device,
                             learning_rate=args.i_learning_rate,
                             I_action_gumbel_max_tau=args.I_action_gumbel_max_tau,
                             I_ubc_gumbel_max_tau=args.I_ubc_gumbel_max_tau,
                             pseudo_ucb_coef=args.pseudo_ucb_coef
                             )
    e_agent = ExtrinsicQAgent(envs,
                              device,
                              args.tau,
                              learning_rate=args.e_learning_rate,
                              E_action_gumbel_max_tau=args.E_action_gumbel_max_tau)
    dim_x = 70
    dim_y = 53

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        dim_x=dim_x,
        dim_y=dim_y,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )

    from collections import deque

    ep_info_buffer = deque(maxlen=1000)
    ep_success_buffer = deque(maxlen=1000)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # obs, _ = envs.reset(seed=args.seed)

    intrinsic_reward_statistics = 0
    intrinsic_reward_statistics_count = 0
    total_env_step_call = 0
    global_i_step = 0

    c_hn_size = (i_agent.byol.num_layers, args.num_envs, i_agent.byol.hidden_size)
    last_ep_c_hn = torch.ones(c_hn_size)  # (self.num_layers, args.num_envs , self.hidden_size)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if (global_step % 100) / 100 > epsilon:
            current_agent = e_agent
            e_agent.rollout_step(args,device,dim_x,dim_y,envs,ep_info_buffer, ep_success_buffer,rb,global_step)
        else:
            current_agent = i_agent
            last_ep_c_hn = i_agent.rollout_step(args,device,dim_x,dim_y,envs,ep_info_buffer, ep_success_buffer,rb,last_ep_c_hn)
            if global_i_step % args.byol_update_embedding_to_target_freq == 0:
                i_agent.byol.embedding.update_target()
            global_i_step += 1

        total_env_step_call += args.num_envs * args.num_steps

        if global_step % args.log_interval == 0:
            assert ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
            if len(ep_info_buffer) > 0:
                print("global_step", global_step, ",   total_env_step_call", total_env_step_call, ",   I_rew",
                      intrinsic_reward_statistics)
                intrinsic_reward_statistics = 0
                intrinsic_reward_statistics_count = 0
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
