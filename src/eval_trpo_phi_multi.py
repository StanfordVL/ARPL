"""

This module is the main evaluation module for TRPO agents.
It does a series of rollouts by testing on ALL conifgurations.

"""

from rllab.envs.gym_env import GymEnv
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout

import numpy as np
import argparse
from phi_config import phi_configs
from curriculum_config import curriculum_configs
from environments import dynamic_environments, original_environments


# number of rollouts (trajectories) to do
num_rollouts = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('num_agents', metavar='num_agents', type=int,
                        help='number of agents, to do rollout over multiple trained agents of same type')

    args = parser.parse_args()

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

    # the number of agents
    num_agents = args.num_agents

    # store rollout results in these arrays
    mean_rewards = np.zeros((len(curriculum_configs) + 1, len(phi_configs)))
    std_rewards = np.zeros((len(curriculum_configs) + 1, len(phi_configs)))

    # NOTE: the first train configuration is the nominal policy, an adversarial learner

    # iterate over train configurations (each is a curriculum learner)
    for train_config_num in range(len(curriculum_configs) + 1):

        # iterate over test configurations
        for test_config_num, test_config in enumerate(phi_configs):
            print("train config num : {}".format(train_config_num))
            print("test config num : {}".format(test_config_num))

            rollouts = []

            # iterate over agents
            for agent_num in range(num_agents):

                real_config_num = train_config_num - 1
                if train_config_num == 0:
                    real_config_num = "nominal"

                file_str = '../policies/{}/policy_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind],
                                                                                dynamic_environments[args.env_ind],
                                                                                real_config_num,
                                                                                agent_num)

                # read in the agent's policy
                policy = loadModel(file_str)

                if train_config_num == 0:
                    # set configuration for nominal policy
                    policy.set_config(test_config)
                    curriculum = None
                else:
                    # note that policy config is set through the curriculum
                    # by having only one element, we ensure this is the config during rollouts
                    curriculum = [test_config]

                cum_rewards = []
                for i in range(num_rollouts):
                    rollout_dict = rollout(env=env,
                                           agent=policy,
                                           max_path_length=env.horizon,
                                           curriculum=curriculum)
                    cum_rewards.append(np.sum(rollout_dict["rewards"]))
                rollouts.append(cum_rewards)

            mean_rewards[train_config_num][test_config_num] = np.mean(rollouts)
            std_rewards[train_config_num][test_config_num] = np.std(rollouts)

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




