"""

This module is the main evaluation module for TRPO agents.
It does a series of rollouts by testing on ALL conifgurations.

"""

from rllab.envs.gym_env import GymEnv
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout
from rllab.policies.curriculum_policy import CurriculumPolicy

import numpy as np
import argparse
from curriculum_config import curriculum_configs
from environments import dynamic_environments
from read_rollouts import read_rollout
from curriculum_config import get_ddpg_curriculum_configs_cartpole
from phi_config import get_all_phi_configs_ddpg
from multiprocessing import Pool, Queue, Manager
import time

import os

def format_file_name(data):
    experiment_name, train_config_num, test_config_num, env_ind, agent_num, cum_rewards = data
    return "exp_{}_env_{}_train_{}_test_{}_agent_{}.pickle".format(experiment_name, env_ind, train_config_num, test_config_num, agent_num)

def rollout_one(ckpt_path, 
                eval_path, 
                experiment_name, 
                train_config_num, 
                test_configs, 
                test_config_num, 
                env_ind, 
                agent_num,
                q):
    print("[env: {}, train: {}, test: {}, agent: {}]: Start".format(env_ind, train_config_num, test_config_num, agent_num))
    env = GymEnv(dynamic_environments[args.env_ind])
    test_config = test_configs[test_config_num]

    # iterate over test configurations
    # print("train config num : {}".format(train_config_num))
    # print("test config num : {}".format(test_config_num))
    # print("env_ind: {}".format(env_ind))
    # print("agent num: {}".format(agent_num))

    fname = 'agent_{}_config_{}_agent_{}'.format(
        dynamic_environments[env_ind],
        train_config_num,
        agent_num)
    file_str = os.path.join(ckpt_path, fname)

    # read in the agent's policy
    policy, qf = loadModel(file_str)
    curriculum = [test_config]

    cum_rewards = []
    for i in range(num_rollouts):
        rollout_dict = rollout(env=env,
                               agent=policy,
                               max_path_length=env.horizon,
                               curriculum=curriculum)
        cum_rewards.append(np.sum(rollout_dict["rewards"]))
    data_row = (experiment_name, train_config_num, test_config_num, env_ind, agent_num, cum_rewards)
    q.put(data_row)
    saveModel(data_row, os.path.join(eval_path, format_file_name(data_row)))
    print("[env: {}, train: {}, test: {}, agent: {}]: Done".format(env_ind, train_config_num, test_config_num, agent_num))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate your very own DDPG agent!')
    parser.add_argument('experiment_name', metavar='experiment_name', type=str, 
                        help='path of all checkpoint files')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('num_agents', metavar='num_agents', type=int, default='1',
                        help='number of agents, to do rollout over multiple trained agents of same type')
    parser.add_argument('--ckpt_path', metavar='ckpt_path', type=str, default='',
                        help='path of all checkpoint files')
    parser.add_argument('--eval_path', metavar='eval_path', type=str, default='',
                        help='path of all checkpoint files')
    parser.add_argument('--num_workers', metavar='num_workers', type=int, default=1,
                        help='number of workers to evaluate')
    
    args = parser.parse_args()
    if args.ckpt_path == '':
        args.ckpt_path = 'ckpt/{}'.format(args.experiment_name)
    if args.eval_path == '':
        args.eval_path = 'eval/{}'.format(args.experiment_name)

    os.makedirs(args.eval_path, exist_ok=True)

    num_rollouts = 100
    train_configs = get_ddpg_curriculum_configs_cartpole()
    test_configs = get_all_phi_configs_ddpg()
    p = Pool(args.num_workers)
    res_coll = []
    q = Manager().Queue()

    for train_config_num in range(len(train_configs)):
        for test_config_num in range(len(test_configs)):
            for agent_num in range(args.num_agents):
                res = p.apply_async(rollout_one, 
                    (args.ckpt_path, 
                    args.eval_path, 
                    args.experiment_name, 
                    train_config_num, 
                    test_configs, 
                    test_config_num, 
                    args.env_ind,
                    agent_num,
                    q))
                res_coll.append(res)
    for res in res_coll:
        res.get()

    all_outputs = []
    summarized_outputs = []
    while not q.empty():
        output = q.get()
        all_outputs.append(output)
        experiment_name, train_config_num, test_config_num, env_ind, agent_num, rollouts = output
        summarized_outputs.append((experiment_name, train_config_num, test_config_num, env_ind, agent_num, np.mean(rollouts), np.std(rollouts)))

    saveModel(all_outputs, os.path.join(args.eval_path, 'all_data.pickle'))
    saveModel(summarized_outputs, os.path.join(args.eval_path, 'summary_data.pickle'))




