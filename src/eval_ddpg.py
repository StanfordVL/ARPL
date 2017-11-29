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
from phi_config import all_phi_configs_ddpg
from curriculum_config import curriculum_configs
from environments import dynamic_environments
from read_rollouts import read_rollout

from multiprocessing import Process, Queue
import time

import os

# number of rollouts (trajectories) to do
num_rollouts = 100


def rollout_row(train_config_num, env_ind, env, q, path):

    mean_rollouts = np.zeros(len(all_phi_configs_ddpg))
    std_rollouts = np.zeros(len(all_phi_configs_ddpg))

    # iterate over test configurations
    for test_config_num, test_config in enumerate(all_phi_configs_ddpg):
        print("train config num : {}".format(train_config_num))
        print("test config num : {}".format(test_config_num))

        rollouts = []

        # iterate over agents
        for agent_num in range(num_agents):
            fname = 'agent_{}_config_{}_agent_{}'.format(
                dynamic_environments[env_ind],
                train_config_num,
                agent_num)
            file_str = os.path.join(path, fname)

            # read in the agent's policy
            policy, qf = loadModel(file_str)

            if train_config_num == 0:
                # set configuration for nominal policy
                policy.set_config(test_config)
                curriculum = None
            else:
                # note that policy config is set through the curriculum
                # by having only one element, we ensure this is the config during rollouts
                # assert(isinstance(policy, CurriculumPolicy))
                curriculum = [test_config]

            cum_rewards = []
            for i in range(num_rollouts):
                rollout_dict = rollout(env=env,
                                       agent=policy,
                                       max_path_length=env.horizon,
                                       curriculum=curriculum)
                cum_rewards.append(np.sum(rollout_dict["rewards"]))
            rollouts.append(cum_rewards)

        mean_rollouts[test_config_num] = np.mean(rollouts)
        std_rollouts[test_config_num] = np.std(rollouts)
        q.put((train_config_num, test_config_num, mean_rollouts[test_config_num], std_rollouts[test_config_num]))

    # write to file in case something weird with multiproc happens...
    saveModel([mean_rollouts, std_rollouts],
              'rollouts_{}_config_{}'.format(dynamic_environments[env_ind], train_config_num))

    print("GOT HERE {}".format(train_config_num))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate your very own DDPG agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('num_agents', metavar='num_agents', type=int,
                        help='number of agents, to do rollout over multiple trained agents of same type')
    parser.add_argument('ckpt_path', metavar='ckpt_path', type=str,
                        help='path of all checkpoint files')
    args = parser.parse_args()

    env_ind = args.env_ind
    num_agents = args.num_agents
    ckpt_path = args.ckpt_path
    train_config_nums = range(9)
    # train_config_nums = [5,6,7,8,9,10,11,12]
    


    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

    # store rollout results in these arrays
    mean_rewards = np.zeros((len(train_config_nums), len(all_phi_configs_ddpg)))
    std_rewards = np.zeros((len(train_config_nums), len(all_phi_configs_ddpg)))

    # iterate over train configurations
    processes = []
    q = Queue()
    for train_config_num in train_config_nums:
        p = Process(target=rollout_row, args=(train_config_num, env_ind, env, q, ckpt_path))
        processes.append(p)
        p.start()
        time.sleep(1)

    ind = 0
    for p in processes:
        print('waiting on {}'.format(ind))
        p.join()
        ind += 1

    while not q.empty():
        train_config_num, test_config_num, rollouts_mean, rollouts_std = q.get()
        mean_rewards[train_config_num - 5][test_config_num] = rollouts_mean
        std_rewards[train_config_num - 5][test_config_num] = rollouts_std

    # assert that reconstruction matches full table
    # mean_red, std_red = read_rollout(".", dynamic_environments[args.env_ind],
    #                                  read_multi=True, num_files=len(curriculum_configs)+1)
    # for i in range(len(curriculum_configs) + 1):
    #     for j in range(len(all_phi_configs_ddpg)):
    #         assert(mean_red[i][j] == mean_rewards[i][j])
    #         assert(std_red[i][j] == std_red[i][j])


    print("")
    print("mean_rewards")
    print("")
    print(mean_rewards)
    print("")
    print("std_rewards")
    print("")
    print(std_rewards)

    saveModel([mean_rewards, std_rewards],
              'rollouts_{}'.format(dynamic_environments[args.env_ind]))




