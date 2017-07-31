"""

This module is the main evaluation module for TRPO agents.
It does a series of rollouts by testing on ALL conifgurations.

"""

# TODO: ask about types of rollouts we want, also learning curve stuff

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout

import numpy as np
import argparse
from phi_config import phi_configs
from config import configs
from environments import dynamic_environments, original_environments

# number of samplers to use during rollout
num_workers = 4

# number of rollouts (trajectories) to do
num_rollouts = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('config_num', metavar='config_num', type=int,
                        help='configuration number to evaluate (see config.py)')
    parser.add_argument('num_agents', metavar='num_agents', type=int,
                        help='number of agents, to do rollout over multiple trained agents of same type')

    args = parser.parse_args()

    # get the original state space size first
    org_env = GymEnv(original_environments[args.env_ind])
    org_env_size = org_env.observation_space.shape[0]
    org_env.terminate()

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

    # the configuration settings
    eval_config = configs[args.config_num]

    if args.env_ind == 0:
        # batch size for Inverted Pendulum
        eval_config.batch_size = 5000
    else:
        # batch size for all other environments
        eval_config.batch_size = 25000

    # the number of agents
    num_agents = args.num_agents


    # store rollout results in these arrays
    mean_rewards = np.zeros(len(phi_configs))
    std_rewards = np.zeros(len(phi_configs))

    # iterate over test configurations
    for test_config_num, test_config in enumerate(phi_configs):
        print("test config num : {}".format(test_config_num))

        rollouts = []

        # iterate over agents
        for agent_num in range(num_agents):

            file_str = '../policies/{}/policy_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind],
                                                                            dynamic_environments[args.env_ind],
                                                                            args.config_num, agent_num)

            # read in the agent's policy
            policy = loadModel(file_str)

            # set policy parameters to ensure we test correctly (these are used by the rollout function internally)
            policy.adversarial = test_config.adversarial
            policy.eps = test_config.eps
            policy.probability = test_config.probability
            policy.use_dynamics = test_config.use_dynamics
            policy.random = test_config.random
            policy.observable_noise = test_config.observable_noise
            policy.use_max_norm = test_config.use_max_norm

            cum_rewards = []
            for i in range(num_rollouts):
                rollout_dict = rollout(env=env,
                                       agent=policy,
                                       max_path_length=env.horizon)
                cum_rewards.append(np.sum(rollout_dict["rewards"]))
            rollouts.append(cum_rewards)

        mean_rewards[test_config_num] = np.mean(rollouts)
        std_rewards[test_config_num] = np.std(rollouts)

    print("mean_rewards")
    print(mean_rewards)
    print("std_rewards")
    print(std_rewards)

    saveModel([mean_rewards, std_rewards],
              'rollouts_{}_config_{}'.format(dynamic_environments[args.env_ind], args.config_num))



