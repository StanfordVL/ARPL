from environments import dynamic_environments
from make_table import print_table_normal, init_doc, end_doc
import numpy as np
from read_rollouts import read_single_rollout, read_single_rollout_agent
import matplotlib.pyplot as plt
from phi_config import phi_configs
from curriculum_config import curriculum_configs

if __name__ == '__main__':

    # set this to True if you need to aggregate rollout results across multiple agents
    agent = True  #False
    num_agents = 30

    num_rows = 4  # len(curriculum_configs) + 1

    path = "../rollouts_curriculum_new_adv"

    col_headers = ["phi", "nom"]
    for phi_config in phi_configs[1:]:
        col_headers.append("{}".format(round(phi_config.probability, 2)))
    col_headers.append("mean")

    row_headers = ["nom"]
    for i in range(1, num_rows):
        row_headers.append("{}".format(i - 1))

    init_doc("Phi Scan Results, Adversarial Dynamics Testing")

    for env in dynamic_environments[:1]:
        mean_rewards = []
        std_rewards = []

        for i in range(num_rows):
            if agent:
                mr, stdr = read_single_rollout_agent(path, env, i, num_agents)
            else:
                mr, stdr = read_single_rollout(path, env, i)
            assert (len(phi_configs) == len(mr))

            mr = list(mr)
            stdr = list(stdr)

            # add in row-means
            mean_mr = np.mean(mr)
            mr.append(mean_mr)
            mean_stdr = np.mean(stdr)
            stdr.append(mean_stdr)

            mean_rewards.append(mr)
            std_rewards.append(stdr)

        # caption for table goes here
        caption = env + " mean rewards"

        print_table_normal(mean_rewards, col_headers, row_headers, caption)

        caption = env + " std dev rewards"
        print_table_normal(std_rewards, col_headers, row_headers, caption)

    end_doc()


