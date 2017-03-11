import argparse
from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Runs DDPG cartpole experiment')
    parser.add_argument('--qf-tv-reg', type=float, help='Total variation regularisation coefficient for the q function')
    parser.add_argument('--qf-weight-decay', type=float, help='Weight decay coefficient for the q-function')
    return parser.parse_args()

def make_run_task(args):
    def run_task(*_):
        env = normalize(CartpoleEnv())

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32)
        )

        es = OUStrategy(env_spec=env.spec)

        qf = ContinuousMLPQFunction(env_spec=env.spec)

        algo = DDPG(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            batch_size=32,
            max_path_length=100,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=1000,
            discount=0.99,
            scale_reward=0.01,
            qf_learning_rate=1e-3,
            qf_tv_reg=args.qf_tv_reg,
            qf_weight_decay=args.qf_weight_decay,
            policy_learning_rate=1e-4,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train()
    return run_task

def main(args):
    run_task = make_run_task(args)
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
