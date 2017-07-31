"""

This module records videos for a given collection of agents.

"""

from rllab.envs.gym_env import GymEnv
from rllab.policies.save_policy import saveModel, loadModel
from rllab.sampler.utils import rollout

import numpy as np
from environments import dynamic_environments, original_environments
from phi_config import phi_configs
from glob import glob
import argparse

# glob pattern to match on
GLOB_PATTERN = "../policies/policy_*"

# video directory to store videos in
VIDEO_DIRECTORY = "../videos"

# number of rollouts (trajectories) to do per agent
num_rollouts = 2

if __name__ == '__main__':
    
    # list of policies to evaluate
    glob_list = glob(GLOB_PATTERN)
    num_files = len(glob_list)

    # percentages of original dynamics to evaluate (50% to 150%)
    percentages = np.arange(0.5, 1.51, 0.1) 
    num_param_evals = len(percentages)

    for f_ind, fname in enumerate(glob_list):
        print("FIND : {}".format(f_ind))
        print("FNAME : {}".format(fname))
        env_name = fname.split('/')[-1].split("_")[1]
        f_suffix = "_".join(fname.split('/')[-1].split("_")[1:])
        env_ind = dynamic_environments.index(env_name)

        # get the original state space size first
        org_env = GymEnv(original_environments[env_ind])
        org_env_size = org_env.observation_space.shape[0]
        org_env.terminate()

        # the environment
        env = GymEnv(dynamic_environments[env_ind], log_dir=VIDEO_DIRECTORY)

        # rollout results
        results = np.zeros((num_param_evals, num_param_evals))

        # read in the agent's policy
        policy = loadModel(fname)

        o = env.reset()
        original_dynamics = o[org_env_size:]
        assert(len(original_dynamics) == 2)

        for i in range(num_param_evals):
            for j in range(num_param_evals):
                new_dynamics = original_dynamics.copy()
                new_dynamics[0] = percentages[i] * original_dynamics[0]
                new_dynamics[1] = percentages[j] * original_dynamics[1]
                policy.set_dynamics = new_dynamics
                policy.adversarial = False

                # curriculum is just nominal config, no adversarial
                curriculum = [phi_configs[0]]

                # average over several rollouts
                cum_rewards = np.zeros(num_rollouts)
                for k in range(num_rollouts):
                    rollout_dict = rollout(env=env,
                                           agent=policy,
                                           max_path_length=env.horizon,
                                           curriculum=curriculum)
                    cum_rewards[k] = np.sum(rollout_dict["rewards"])
                results[i, j] = np.mean(cum_rewards)

        saveModel(results, "epopt_{}".format(f_suffix))
