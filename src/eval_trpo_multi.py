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
    parser.add_argument('num_agents', metavar='num_agents', type=int,
                        help='number of agents, to do rollout over multiple trained agents of same type')

    args = parser.parse_args()

    # get the original state space size first
    org_env = GymEnv(original_environments[args.env_ind])
    org_env_size = org_env.observation_space.shape[0]
    org_env.terminate()

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

    # the number of agents
    num_agents = args.num_agents

    # store rollout results in these arrays
    mean_rewards = np.zeros((len(configs), len(configs)))
    std_rewards = np.zeros((len(configs), len(configs)))

    # iterate over train configurations
    for train_config_num, eval_config in enumerate(configs):

        if args.env_ind == 0:
            # batch size for Inverted Pendulum
            eval_config.batch_size = 5000
        else:
            # batch size for all other environments
            eval_config.batch_size = 25000

        # iterate over test configurations
        for test_config_num, test_config in enumerate(configs):
            print("train config num : {}".format(train_config_num))
            print("test config num : {}".format(test_config_num))

            rollouts = []

            # iterate over agents
            for agent_num in range(num_agents):

                file_str = '../policies/{}/policy_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind], dynamic_environments[args.env_ind], train_config_num, agent_num)

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

                # # use TRPO algorithm's sampler to sample a bunch of trajectories under the policy above
                #
                # baseline = LinearFeatureBaseline(env_spec=env.spec)
                #
                # algo = TRPO(
                #     env=env,
                #     policy=policy,
                #     baseline=baseline,
                #     batch_size=num_rollouts, ### defines number of trajectories to collect
                #     max_path_length=env.horizon,
                #     n_itr=test_config.num_iter,
                #     discount=test_config.discount,
                #     step_size=test_config.step_size,
                #     gae_lambda=test_config.gae_lambda,
                #     num_workers=num_workers,
                #     use_num_paths=True, ### ensures that we collect @num_rollouts number of trajectories
                #     plot_learning_curve=test_config.plot_learning_curve,
                #     trial=agent_num,
                # )
                #
                # algo.start_worker()
                # paths = algo.sampler.obtain_samples(0)
                # algo.shutdown_worker()

                cum_rewards = []
                for i in range(num_rollouts):
                    rollout_dict = rollout(env=env,
                                           agent=policy,
                                           max_path_length=env.horizon)
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




