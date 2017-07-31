from six.moves import cPickle
from environments import dynamic_environments
import matplotlib.pyplot as plt
import numpy as np


def read_single_rewards(path, env, config_num, agent_num):
    file_str = "{}/{}/rewards_{}_config_{}_agent_{}".format(path, env, env, config_num, agent_num)
    f = open(file_str, 'rb')
    itr_lst, mr, stdr = cPickle.load(f)
    f.close()
    return np.array(itr_lst), np.array(mr), np.array(stdr)

def plot_rewards(itr_lst, mr, stdr, save_path):
    plt.figure()
    plt.plot(itr_lst, mr, 'r')
    #plt.plot(itr_lst, mr + stdr, 'b')
    #plt.plot(itr_lst, mr - stdr, 'b')
    plt.title('Learning Curve')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':

    path = "../policies_curriculum_new_adv"
    num_files = 4 #21 #19
    num_agents = 30 #3

    for env in dynamic_environments[:1]:
        base_save_path = "../rewards_curriculum_new_adv/{}".format(env)
        for i in range(num_files):
            ind = i
            # if i == 0:
            #     ind = "nominal"
            # else:
            #     ind = i - 1
            for j in range(num_agents):
                itr_lst, mr, stdr = read_single_rewards(path, env, ind, j)
                save_path = "{}/rewards_{}_config_{}_agent_{}.png".format(base_save_path, env, ind, j)
                plot_rewards(itr_lst, mr, stdr, save_path)




