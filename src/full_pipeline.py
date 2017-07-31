"""

This module represents the full pipeline for training and evaluating agents.

"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.policies.curriculum_policy import CurriculumPolicy
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout

from rllab.envs.normalized_env import normalize

import argparse
import numpy as np
from curriculum_config import curriculum_configs
# from curriculum_config_1 import curriculum_configs
from phi_config import all_phi_configs
from environments import dynamic_environments, original_environments

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
from make_table import print_table_normal, init_doc, end_doc

# quick run for debugging
DEBUG = False

num_rollouts = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('config_num', metavar='config_num', type=int,
                        help='configuration number (see curriculum_config.py)')
    parser.add_argument('agent_num', metavar='agent_num', type=int,
                        help='agent number (for file writing purposes)')

    args = parser.parse_args()

    # get the original state space size first
    org_env = GymEnv(original_environments[args.env_ind])
    org_env_size = org_env.observation_space.shape[0]
    org_env.terminate()

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])
    #env = normalize(GymEnv(dynamic_environments[args.env_ind]), normalize_obs=True)

    # the configuration settings
    curriculum_config = curriculum_configs[args.config_num]

    if args.env_ind == 0:
        # batch size for Inverted Pendulum
        curriculum_config.set_batch_size(5000)
    else:
        # batch size for all other environments
        curriculum_config.set_batch_size(25000)

    # the nominal config
    config = curriculum_config.curriculum_list[0]

    # the agent number
    agent_num = args.agent_num

    # define policy by reading from config class
    policy = CurriculumPolicy(
        env_spec=env.spec,
        hidden_sizes=config.hidden_sizes,
        adaptive_std=config.adaptive_std,
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
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if DEBUG:
        n_itr = 5
    else:
        n_itr = config.num_iter

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=config.batch_size,
        max_path_length=env.horizon,
        n_itr=n_itr,
        discount=config.discount,
        step_size=config.step_size,
        gae_lambda=config.gae_lambda,
        num_workers=config.num_workers,
        plot_learning_curve=config.plot_learning_curve,
        trial=agent_num,
    )
    avg_rewards, std_rewards = algo.train()

    print("training completed!")
    saveModel(algo.policy,
              'policy_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))

    # save rewards per model over the iterations
    # also plot the rewards
    if config.plot_learning_curve:
        saveModel([range(n_itr), avg_rewards, std_rewards],
                  'rewards_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))
        
        plt.figure()
        plt.plot(range(n_itr), avg_rewards)
        plt.title('Learning Curve')
        plt.savefig('mr_{}_config_{}_agent_{}.png'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))
        plt.close()

        plt.figure()
        plt.plot(range(n_itr), std_rewards)
        plt.title('Learning Curve')
        plt.savefig('stdr_{}_config_{}_agent_{}.png'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))
        plt.close()

    print("rolling out....")

    if DEBUG:
        num_rollouts = 1

    all_mean_rewards = []
    all_std_rewards = []

    for phi_configs in all_phi_configs:

        # store rollout results in these arrays
        mean_rewards = []
        std_rewards = []

        # iterate over test configurations
        for test_config_num, test_config in enumerate(phi_configs):
            print("test config num : {}".format(test_config_num))

            if algo.policy.mask_augmentation and test_config.use_dynamics and not test_config.random:
                print("invalid config for no augmentation")
                mean_rewards.append(-1.0)
                std_rewards.append(-1.0)
                continue

            curriculum = [test_config]

            cum_rewards = []
            for i in range(num_rollouts):
                rollout_dict = rollout(env=env,
                                       agent=algo.policy,
                                       max_path_length=env.horizon,
                                       curriculum=curriculum)
                cum_rewards.append(np.sum(rollout_dict["rewards"]))

            mean_rewards.append(cum_rewards)
            std_rewards.append(cum_rewards)

        all_mean_rewards.append(mean_rewards)
        all_std_rewards.append(std_rewards)

    real_mean_rewards = []
    real_std_rewards = []

    for mean_rewards, std_rewards in zip(all_mean_rewards, all_std_rewards):

        print("")
        print("mean_rewards")
        print("")
        print(mean_rewards)
        print("")
        print("std_rewards")
        print("")
        print(std_rewards)

        mr = list(map(lambda x : np.mean(x), mean_rewards))
        stdr = list(map(lambda x : np.std(x), std_rewards))

        # add in row-means
        mean_mr = np.mean(mr)
        mr.append(mean_mr)
        mean_stdr = np.mean(stdr)
        stdr.append(mean_stdr)

        real_mean_rewards.append(mr)
        real_std_rewards.append(stdr)

    saveModel([all_mean_rewards, all_std_rewards],
              'rollouts_{}_config_{}_agent{}'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))

    # print rollout results to a latex file

    col_headers = ["phi", "nom"]
    for phi_config in all_phi_configs[0][1:]:
        col_headers.append("{}".format(round(phi_config.probability, 2)))
    col_headers.append("mean")

    row_headers = []
    for i in range(len(all_mean_rewards)):
        row_headers.append(str(i + 1))

    sys.stdout = open('results_{}_config_{}_agent{}.tex'.format(dynamic_environments[args.env_ind], args.config_num, agent_num), 'w')
    init_doc("Phi Scan Results")

    # caption for table goes here
    caption = "Configuration {} evaluated on 4 different test cases.".format(args.config_num)
    print_table_normal(real_mean_rewards, col_headers, row_headers, caption)

    end_doc()


