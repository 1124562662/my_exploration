import argparse
import os
from distutils.util import strtobool


def get_test_arg():
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

    parser.add_argument("--num-steps", type=int, default=128,
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

    parser.add_argument("--net_num", type=int, default=3,
                        help=" ")
    parser.add_argument("--train_net_num", type=int, default=1,
                        help="")

    parser.add_argument("--rnd_train_num", type=int, default=40,
                        help=" ")
    parser.add_argument("--rnd_update_epochs", type=int, default=1,
                        help=" ")
    parser.add_argument("--novelD-alpha", type=float, default=0.5,
                        help="novelD-alpha if positive, do not use novelD if zero")
    parser.add_argument("--extrinsic_rewards_for_i_agent", type=float, default=1,
                        help=" ")
    parser.add_argument("--use-contextual-bandit-UBC", type=bool, default=False,
                        help=" ")  # TODO
    args = parser.parse_args()

    return args