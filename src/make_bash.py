# this file provides helper functions to generate bash scripts for instantiating many screens on a machine

import argparse

def screen_command(name, script, args):
    """

    :param name: screen name
    :param script: script name
    :param args: list of command-line arguments
    :return:
    """

    args_string = ' '.join(args)
    print("screen -dmS {} bash -c \'python {} {}\'".format(name, script, args_string))
    print("echo \'sleeping...\'")
    print("sleep 2")

def make_bash_file(train, env_ind, config_num_lb, config_num_ub, agent_lb, 
                   agent_ub, parallel_agents, use_curriculum):
    if train:
        script_name = "train_trpo"

        if use_curriculum:
            script_name += "_curriculum"
        if not parallel_agents:
            script_name += "_multi"
        script_name += ".py"
    else:
        script_name = "eval_trpo"

        if use_curriculum:
            script_name += "_phi"
        if parallel_agents:
            script_name += "_agent"
        script_name += ".py"

    num_agents = agent_ub - agent_lb

    # initialize bash file
    print("#!/bin/bash")
    print("")

    screen_ind = 0
    for ind in range(config_num_lb, config_num_ub):
        if parallel_agents:
            for agent in range(agent_lb, agent_ub):
                arg_list = [str(env_ind), str(ind), str(agent)]
                screen_name = "screen_{}".format(screen_ind)
                screen_command(name=screen_name, script=script_name, args=arg_list)
                screen_ind += 1
        else:
            arg_list = [str(env_ind), str(ind), str(num_agents)]
            screen_name = "screen_{}".format(screen_ind)
            screen_command(name=screen_name, script=script_name, args=arg_list)
            screen_ind += 1

if __name__ == '__main__':

    train = True

    parser = argparse.ArgumentParser(description='Evaluate your very own TRPO agent!')
    parser.add_argument('env_ind', metavar='env_ind', type=int,
                        help='index corresponding to environment name (see environments.py)')
    parser.add_argument('config_num_lb', metavar='config_num_lb', type=int,
                        help='configuration number (see config.py)')
    parser.add_argument('config_num_ub', metavar='config_num_ub', type=int,
                        help='configuration number (see config.py)')
    parser.add_argument('agent_lb', metavar='agent_lb', type=int,
                        help='lower agent number')
    parser.add_argument('agent_ub', metavar='agent_ub', type=int,
                        help='upper agent number')
    parser.add_argument('parallel_agents', metavar='parallel_agents', type=int,
                        help='if true, use separate screens per agent')
    parser.add_argument('use_curriculum', metavar='use_curriculum', type=int,
                        help='whether to use curriculum learning or not')

    args = parser.parse_args()

    env_ind = args.env_ind
    config_num_lb = args.config_num_lb
    config_num_ub = args.config_num_ub
    agent_lb = args.agent_lb
    agent_ub = args.agent_ub
    parallel_agents = args.parallel_agents
    use_curriculum = args.use_curriculum

    make_bash_file(train=train, env_ind=env_ind, config_num_lb=config_num_lb, 
                   config_num_ub=config_num_ub, agent_lb=agent_lb, agent_ub=agent_ub,
                   parallel_agents=parallel_agents, use_curriculum=use_curriculum)

    # if train:
    #     script_name = "train_trpo"

    #     if use_curriculum:
    #         script_name += "_curriculum"
    #     if not parallel_agents:
    #         script_name += "_multi"
    #     script_name += ".py"
    # else:
    #     script_name = "eval_trpo"

    #     if use_curriculum:
    #         script_name += "_phi"
    #     if parallel_agents:
    #         script_name += "_agent"
    #     script_name += ".py"

    # # initialize bash file
    # print("#!/bin/bash")
    # print("")

    # screen_ind = 0
    # for ind in range(lb, ub):
    #     if parallel_agents:
    #         for agent in range(a_lb, a_ub):
    #             arg_list = [str(env_ind), str(ind), str(agent)]
    #             screen_name = "screen_{}".format(screen_ind)
    #             screen_command(name=screen_name, script=script_name, args=arg_list)
    #             screen_ind += 1
    #     else:
    #         arg_list = [str(env_ind), str(ind), str(num_agents)]
    #         screen_name = "screen_{}".format(screen_ind)
    #         screen_command(name=screen_name, script=script_name, args=arg_list)
    #         screen_ind += 1
