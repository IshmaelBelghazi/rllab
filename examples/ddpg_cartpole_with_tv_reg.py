#!/usr/bin/env python
import argparse
import os
import json
from collections import OrderedDict

import pandas as pd
import numpy.random as npr


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
    parser.add_argument('--qf-tv-reg', type=float, default=0., help='Total variation regularisation coefficient for the q function')
    parser.add_argument('--max-tv', type=float, default=5., help='Maximum total variation')
    parser.add_argument('--qf-weight-decay', default=0., type=float, help='Weight decay coefficient for the q-function')
    parser.add_argument('--log-dir', type=str, default=None, help='Path to log dir')
    parser.add_argument('--nexps', type=int, default=3, help='Number of experiments')
    parser.add_argument('--n-epochs', type=int, default=20, help='numberof epochs per experiment')
    parser.add_argument('--nseeds', type=int, default=1, help='number of seeds per experiment')
    parser.add_argument('--seed', type=int, default=1, help='numpy random seed')
    parser.add_argument('--plot', action='store_true', help='plot policy performance')
    parser.add_argument('--nparallel', type=int, default=1, help='number of parallel workers')
    return parser.parse_args()

def dump_args(args, exp_name):
    with open(os.path.join(args.log_dir, exp_name, 'hyperparams.json'), 'w') as f:
        json.dump(vars(args), f)

def make_exp_name(args):
    return 'cartpole_tvreg_{}_wd_{}_seed_{}'.format(args.qf_tv_reg, args.qf_weight_decay, args.seed)

def make_monitoring_df(log_dir):
    exp_dirs = [d[0] for d in os.walk(log_dir)][1:]
    dfs = []
    hyperparams_names = ['qf_tv_reg', 'qf_weight_decay', 'seed']

    for exp_dir in exp_dirs:
        df = pd.read_csv(os.path.join(exp_dir, 'experiment_monitoring.csv'))
        with open(os.path.join(exp_dir, 'hyperparams.json'), 'r') as f:
            hyperparams_dict = json.load(f)
            hyperparams_vals = [hyperparams_dict[key] for key in hyperparams_names]
        hypers_df = pd.concat([pd.DataFrame(data=OrderedDict(zip(hyperparams_names, hyperparams_vals)), index=range(1))] * len(df.index),
                              ignore_index=True)
        dfs.append(pd.concat([df, hypers_df], axis=1))
    all_dfs_df = pd.concat(dfs, axis=0)
    return all_dfs_df

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
            n_epochs=args.n_epochs,
            discount=0.99,
            scale_reward=0.01,
            qf_learning_rate=1e-3,
            qf_tv_reg=args.qf_tv_reg,
            qf_weight_decay=args.qf_weight_decay,
            policy_learning_rate=1e-4,
            plot=args.plot,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train()
    return run_task

def run_experiments(args):
    HEADER_MSG = """
    =====================================================================================
    Running cartpole experiment with: Weight decay = {}, TV regularization: {}. Seed: {}'
    =====================================================================================
    """.format(args.qf_weight_decay, args.qf_tv_reg, args.seed)
    print(HEADER_MSG)
    run_task_fun = make_run_task(args)
    exp_name = make_exp_name(args)
    os.mkdir(os.path.join(args.log_dir, exp_name))
    dump_args(args, exp_name)  # serialze to json
    run_experiment_lite(
        run_task_fun,
        # Number of parallel workers for sampling
        n_parallel=args.nparallel,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=args.seed,
        plot=args.plot,
        log_dir=os.path.join(args.log_dir, exp_name),
        exp_name=exp_name,
        tabular_log_file='experiment_monitoring.csv',
    )

if __name__ == '__main__':
    args = parse_arguments()
    tv_coefs = [0.] + [2.**x for x in range(-10, 6)]
    for enumerate in tv_coefs:
        for _ in range(args.nseeds):
            args.seed = npr.randint(2**32)
            args.qf_tv_reg = tv_coefs[n]
            # args.qf_tv_reg = npr.uniform(low=0.0, high=args.max_tv)
            run_experiments(args)
    print('Collecting experiments info')
    experiments_df = make_monitoring_df(args.log_dir)
    experiments_df.to_hdf(os.path.join(args.log_dir, 'experiments_monitoring.h5'), 'w')
