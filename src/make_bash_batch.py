"""
This script should just be used to generate several bash scripts at once
"""

import sys
from make_bash import make_bash_file

env_strs = ["inv_pend", "cheetah", "hopper", "walker"]

if __name__ == '__main__':

    train = True

    for env_ind in range(4):
        for config_num in range(16):
            sys.stdout = open("{}_{}.sh".format(env_strs[env_ind], config_num + 1), "w")
            make_bash_file(train=train, env_ind=env_ind, config_num_lb=config_num, 
                           config_num_ub=config_num + 1, agent_lb=0, agent_ub=15,
                           parallel_agents=True, use_curriculum=True)
