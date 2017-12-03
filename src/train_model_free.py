"""

This module is the main training module for training DDPG agents.

"""

from rllab.algos.trpo import TRPO
from rllab.algos.ddpg import DDPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.policies.model_free_adversarial_policy import ModelFreeAdversarialPolicy
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.deterministic_mlp_curriculum_policy import DeterministicMLPCurriculumPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from rllab.envs.normalized_env import normalize

import argparse
import numpy as np
from curriculum_config import get_ddpg_curriculum_configs_cartpole
from environments import dynamic_environments, original_environments

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
from make_table import print_table_normal, init_doc, end_doc
from multiprocessing import Pool

def train(env_ind, config_num, agent_num, checkpoint_path):
    # get the original state space size first
    print('[env_ind: {}, config_num: {}, agent_num: {}]: starting'.format(env_ind, config_num, agent_num))
    org_env = GymEnv(original_environments[env_ind])
    org_env_size = org_env.observation_space.shape[0]
    org_env.terminate()

    # the environment
    env = GymEnv(dynamic_environments[env_ind])
    curriculum_configs = get_ddpg_curriculum_configs_cartpole()
    # the configuration settings
    curriculum_config = curriculum_configs[config_num]
    # print('config: ')
    # curriculum_config.print()

    if env_ind == 0:
        # batch size for Inverted Pendulum
        curriculum_config.set_batch_size(5000)
    else:
        # batch size for all other environments
        curriculum_config.set_batch_size(25000)

    # the nominal config
    config = curriculum_config.curriculum_list[0]

    # the agent number
    agent_num = agent_num


    qf = ContinuousMLPQFunction(
        env_spec=env.spec,
        action_merge_layer=-2,
        )


    policy = policy = DeterministicMLPCurriculumPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
        adversarial=config.adversarial,
        eps=config.eps,
        probability=config.probability,
        use_dynamics=config.use_dynamics,
        random=config.random,
        observable_noise=config.observable_noise,
        zero_gradient_cutoff=org_env_size,
        use_max_norm=config.use_max_norm,
        curriculum_list=list(curriculum_config.curriculum_list),
        update_freq=curriculum_config.update_freq,
        mask_augmentation=config.mask_augmentation,
        qf=qf,
    )

    es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.3)


    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=config.num_iter,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        plot_learning_curve=config.plot_learning_curve,
        )
    avg_rewards, std_rewards = algo.train()

    print('[env_ind: {}, config_num: {}, agent_num: {}]: training completed!'.format(env_ind, config_num, agent_num))

    saveModel((algo.policy, algo.qf),
             checkpoint_path + '/' + 'model_{}_config_{}_agent_{}'.format(dynamic_environments[env_ind], config_num, agent_num))

    # save rewards per model over the iterations
    # also plot the rewards
    if config.plot_learning_curve:
        saveModel([range(config.num_iter), avg_rewards, std_rewards],
                  checkpoint_path + '/' + 'rewards_{}_config_{}_agent_{}'.format(dynamic_environments[env_ind], config_num, agent_num))
        
        plt.figure()
        plt.plot(range(config.num_iter), avg_rewards)
        plt.title('Learning Curve')
        plt.savefig(checkpoint_path + '/' + 'mr_{}_config_{}_agent_{}.png'.format(dynamic_environments[env_ind], config_num, agent_num))
        plt.close()

        plt.figure()
        plt.plot(range(config.num_iter), std_rewards)
        plt.title('Learning Curve')
        plt.savefig(checkpoint_path + '/' + 'stdr_{}_config_{}_agent_{}.png'.format(dynamic_environments[env_ind], config_num, agent_num))
        plt.close()


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


