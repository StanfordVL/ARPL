from six.moves import cPickle
from environments import dynamic_environments
from make_table import print_table, init_doc, end_doc
import numpy as np


def read_single_rollout_agent(path, env, config_num, num_agents):
    """

    :param path: directory where rollouts are stored
    :param env: name of environment to read in
    :param config_num: which configuration to read
    :param num_agents: how many agents to aggregate
    :return:
    """
    mr = []
    for agent_num in range(num_agents):
        file_str = "{}/{}/rollouts_{}_config_{}_agent{}".format(path, env, env, config_num, agent_num)
        f = open(file_str, 'rb')
        cum_r, _ = cPickle.load(f)
        f.close()
        mr.append(cum_r)
    real_mr = np.zeros(len(mr[0]))
    real_stdr = np.zeros(len(mr[0]))
    for col in range(len(mr[0])):
        tmp_arr = []
        for row in range(len(mr)):
            tmp_arr.append(mr[row][col])
        real_mr[col] = np.mean(tmp_arr)
        real_stdr[col] = np.std(tmp_arr)

    return real_mr, real_stdr

def read_single_rollout(path, env, config_num):
    """

    :param path: directory where rollouts are stored
    :param env: name of environment to read in
    :param config_num: which configuration to read
    :return:
    """
    file_str = "{}/{}/rollouts_{}_config_{}".format(path, env, env, config_num)
    f = open(file_str, 'rb')
    mr, stdr = cPickle.load(f)
    f.close()
    return mr, stdr

# read rollout results from file
def read_rollout(path, env, read_multi, num_files=19):
    """

    :param path: directory where rollouts are stored
    :param env: name of environment to read in
    :param read_multi: if True, read from many rollout files, each containing a single row
    :param num_files: only used if @read_multi is True, the number of rollout rows to combine
    :return:
    """
    if read_multi:
        mean_rewards = []
        std_rewards = []
        for i in range(num_files):
            mr, stdr = read_single_rollout(path, env, i)
            mean_rewards.append(mr)
            std_rewards.append(stdr)
        mean_rewards = np.array(mean_rewards)
        std_rewards = np.array(std_rewards)

    else:
        file_str = "{}/{}/rollouts_{}".format(path, env, env)
        f = open(file_str, 'rb')
        mean_rewards, std_rewards = cPickle.load(f)
        f.close()

    return mean_rewards, std_rewards

# get indices to index into table for the appropriate phi comparisons
def get_inds(phi1, phi2):
    if phi1 == 0.0 and phi2 == 0.1:
        inds = list(range(13))
    elif phi1 == 0.0 and phi2 == 0.5:
        inds = list(range(7)) + list(range(13, 19))
    elif phi1 == 0.1 and phi2 == 0.5:
        inds = [0] + list(range(7, 19))
    else:
        raise NotImplementedError

    return inds

# takes full rollout table, and indexes into it appropriately to recover a subset
def collect_results(mean_rewards, std_rewards, phi1, phi2):

    inds = get_inds(phi1, phi2)

    m_rewards = []
    sd_rewards = []
    for i in inds:
        m = []
        s = []
        for j in inds:
            m.append(mean_rewards[i][j])
            s.append(std_rewards[i][j])
        m_rewards.append(m)
        sd_rewards.append(s)

    return m_rewards, sd_rewards

if __name__ == '__main__':

    # set this to True if rollout results should be read in one row at a time, instead of all at once
    read_multi = False

    # set the environment to read in
    env_ind = 3
    env = dynamic_environments[env_ind]

    # set the title of the latex document
    title = "Results for Walker2d with nominal policy (500 iter) initialization"

    # number of iterations of training
    num_iters = 500

    # set the path to the folder which contains the rollouts
    #path = "../policies_eps_1000_iters/rollouts_eps_0_00001"
    path = "../walker_nominal_init"

    eps = 0.1

    mean_rewards, std_rewards = read_rollout(path, env, read_multi)

    init_doc(title)

    # scan over phi comparisons
    for phi1, phi2 in [(0.0, 0.1), (0.0, 0.5), (0.1, 0.5)]:
        m_rewards, sd_rewards = collect_results(mean_rewards, std_rewards, phi1, phi2)

        #print(mean_rewards)
        print_table(m_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=True)
        print_table(sd_rewards, env_name=env, phi1=phi1, phi2=phi2, eps=eps, num_iters=num_iters, is_mean=False)

    end_doc()



