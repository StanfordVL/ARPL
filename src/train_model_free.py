"""

This module is the main training module for training DDPG agents.

"""

# from rllab.algos.trpo import TRPO
# from rllab.algos.ddpg import DDPG
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.gym_env import GymEnv
# from rllab.policies.model_free_adversarial_policy import ModelFreeAdversarialPolicy
# from rllab.policies.save_policy import saveModel, loadModel
# from rllab.sampler.utils import rollout
# from rllab.exploration_strategies.ou_strategy import OUStrategy
# from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
# from rllab.policies.deterministic_mlp_curriculum_policy import DeterministicMLPCurriculumPolicy
# from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

# from rllab.envs.normalized_env import normalize

import argparse
import numpy as np
from curriculum_config import get_ddpg_curriculum_configs_cartpole
from environments import dynamic_environments, original_environments

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import subprocess
from make_table import print_table_normal, init_doc, end_doc
from multiprocessing import Pool

def train(env_ind, config_num, agent_num, checkpoint_path):
    directory = os.getcwd()
    train_single_script = os.path.join(directory, 'src/train_model_free_single.py', )
    subprocess.run(['python', 
        train_single_script, 
        str(env_ind), 
        str(config_num), 
        str(agent_num),
        str(checkpoint_path)],
        check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    # parser.add_argument('config_num', metavar='config_num', type=int,
    #                     help='configuration number (see curriculum_config.py)')
    parser.add_argument('--agent_num', metavar='agent_num', type=int, default=10,
                        help='number of agents to train (for file writing purposes)')
    parser.add_argument('--checkpoint_path', metavar='checkpoint_path', type=str, default='ckpt/ddpg',
                        help='path to checkpoint (for file writing purposes)')
    parser.add_argument('--num_workers', metavar='number_of_workers', type=int, default=4,
                        help='number of workers for the pool')
    parser.add_argument('--debug', action='store_true', help='shortcut for debugging')

    args = parser.parse_args()
    if args.debug:
        args.num_workers = 1
    print('args', args)
    config_nums = range(len(get_ddpg_curriculum_configs_cartpole()))
    if args.debug:
        # config_nums = range(1)
        config_nums = [len(get_ddpg_curriculum_configs_cartpole()) - 1]
    if args.debug:
        args.checkpoint_path = 'ckpt/debug'


    # iterate over train configurations
    p = Pool(args.num_workers)
    res_coll = []
    for train_config_num in config_nums:
        for agent_num in range(args.agent_num):
            res = p.apply_async(train, (args.env_ind, train_config_num, agent_num, args.checkpoint_path))
            res_coll.append(res)
    for res in res_coll:
        res.get()
    p.close()
    p.join()


