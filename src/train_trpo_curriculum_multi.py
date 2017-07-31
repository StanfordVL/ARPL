"""

This module is the main training module for training TRPO agents.

"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.policies.curriculum_policy import CurriculumPolicy
from rllab.policies.save_policy import saveModel, loadModel

import argparse
from curriculum_config import curriculum_configs
from environments import dynamic_environments, original_environments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('config_num', metavar='config_num', type=int,
                        help='configuration number (see config.py)')
    parser.add_argument('num_agents', metavar='num_agents', type=int,
                        help='number of agents to train')

    args = parser.parse_args()

    # get the original state space size first
    org_env = GymEnv(original_environments[args.env_ind])
    org_env_size = org_env.observation_space.shape[0]
    org_env.terminate()

    # the environment
    env = GymEnv(dynamic_environments[args.env_ind])

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
    num_agents = args.num_agents

    for agent_num in range(num_agents):

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
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=config.batch_size,
            max_path_length=env.horizon,
            n_itr=config.num_iter,
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
        if config.plot_learning_curve:
            saveModel([range(config.num_iter), avg_rewards, std_rewards],
                      'rewards_{}_config_{}_agent_{}'.format(dynamic_environments[args.env_ind], args.config_num, agent_num))
