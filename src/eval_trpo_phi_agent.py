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
    parser.add_argument('config_num', metavar='config_num', type=int,
                        help='configuration number to evaluate (see config.py)')
    parser.add_argument('agent_num', metavar='agent_num', type=int,
                        help='agent number (for file writing purposes)')

    args = parser.parse_args()

    base_dir = "../policies_curriculum_new"

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

    # the agent number
    agent_num = args.agent_num

    # the configuration settings
    train_config_num = args.config_num

    # store rollout results in these arrays
    mean_rewards = []
    std_rewards = []

    # NOTE: the first train configuration is the nominal policy, an adversarial learner

    # iterate over test configurations
    for test_config_num, test_config in enumerate(phi_configs):
        print("train config num : {}".format(train_config_num))
        print("test config num : {}".format(test_config_num))

        real_config_num = train_config_num

        # real_config_num = train_config_num - 1
        # if train_config_num == 0:
        #     real_config_num = "nominal"

        file_str = '{}/{}/policy_{}_config_{}_agent_{}'.format(base_dir,
                                                               dynamic_environments[args.env_ind],
                                                               dynamic_environments[args.env_ind],
                                                               real_config_num,
                                                               agent_num)

        # read in the agent's policy
        policy = loadModel(file_str)

        curriculum = [test_config]

        # if train_config_num == 0:
        #     # set configuration for nominal policy
        #     policy.set_config(test_config)
        #     curriculum = None
        # else:
        #     # note that policy config is set through the curriculum
        #     # by having only one element, we ensure this is the config during rollouts
        #     curriculum = [test_config]

        cum_rewards = []
        for i in range(num_rollouts):
            rollout_dict = rollout(env=env,
                                   agent=policy,
                                   max_path_length=env.horizon,
                                   curriculum=curriculum)
            cum_rewards.append(np.sum(rollout_dict["rewards"]))

        mean_rewards.append(cum_rewards)
        std_rewards.append(cum_rewards)

    print("")
    print("mean_rewards")
    print("")
    print(mean_rewards)
    print("")
    print("std_rewards")
    print("")
    print(std_rewards)

    saveModel([mean_rewards, std_rewards],
              'rollouts_{}_config_{}_agent{}'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))




