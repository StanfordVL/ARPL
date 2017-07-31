from six.moves import cPickle
from environments import dynamic_environments
from make_table import print_table, init_doc, end_doc
import numpy as np
from read_rollouts import read_rollout, collect_results

# set this to True if rollout results should be read in one row at a time, instead of all at once,
# one entry per environment
#read_multis = [False, True, True, True]
read_multis = [False, False, False, False]

# set this to True if reading epsilon scan
read_eps = False

# number of iterations of training
num_iters = 1000

# set the path to the folder which contains the rollouts
#path = "../policies_eps_{}_iters".format(num_iters)
path = "../rollouts_{}_iters".format(num_iters)

# set the title of the LaTex document
title = "Results for {} training iterations".format(num_iters)

if __name__ == '__main__':

    if read_eps:
        # assuming inverted pendulum
        env_ind = 0
        env = dynamic_environments[0]

        init_doc(title)

        for eps in [0.01, 0.001, 0.0001, 0.00001]:

            if eps == 0.00001:
                # avoid scientific notation...
                eps_str = "{:.5f}".format(0.00001)
            else:
                eps_str = str(eps)

            conv = '_'.join(eps_str.split('.'))
            full_path = '{}/rollouts_eps_{}'.format(path, conv)

            # scan over phi comparisons
            for phi1, phi2 in [(0.0, 0.1), (0.0, 0.5), (0.1, 0.5)]:
                mean_rewards, std_rewards = read_rollout(full_path, env, read_multis[env_ind])

                m_rewards, sd_rewards = collect_results(mean_rewards, std_rewards, phi1, phi2)

                # print(mean_rewards)
                print_table(m_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=True)
                print_table(sd_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=False)


        end_doc()

    else:
        eps = 0.1

        init_doc(title)

        # scan over environments
        for env_ind in range(4):
            env = dynamic_environments[env_ind]

            # scan over phi comparisons
            for phi1, phi2 in [(0.0, 0.1), (0.0, 0.5), (0.1, 0.5)]:

                mean_rewards, std_rewards = read_rollout(path, env, read_multis[env_ind])

                m_rewards, sd_rewards = collect_results(mean_rewards, std_rewards, phi1, phi2)

                #print(mean_rewards)
                print_table(m_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=True)
                print_table(sd_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=False)

        end_doc()

