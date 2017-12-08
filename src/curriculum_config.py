"""

Use these configurations to use curriculum learning.

This file contains configurations for experiments with process noise.

"""

import numpy as np
from config import Config
from curriculum_config_org import CurriculumConfig

curriculum_configs = []

# how to do adversarial / random perturbations
use_max_norm = False

# number of iterations
num_iter = 2000 # 1000

# config 0 is adversarial, phi_max=0.05, step/freq=(5,200), eps=0.01
# config 1 is adversarial, phi_max=0.05, step/freq=(5,200), eps=0.1
# config 2 is adversarial, phi_max=0.05, step/freq=(10,100), eps=0.01
# config 3 is adversarial, phi_max=0.05, step/freq=(10,100), eps=0.1
# config 4 is adversarial, phi_max=0.1, step/freq=(5,200), eps=0.01
# config 5 is adversarial, phi_max=0.1, step/freq=(5,200), eps=0.1
# config 6 is adversarial, phi_max=0.1, step/freq=(10,100), eps=0.01
# config 7 is adversarial, phi_max=0.1, step/freq=(10,100), eps=0.1
# config 8 is random, phi_max=0.05, step/freq=(5,200), eps=0.01
# config 9 is random, phi_max=0.05, step/freq=(5,200), eps=0.1
# config 10 is random, phi_max=0.05, step/freq=(10,100), eps=0.01
# config 11 is random, phi_max=0.05, step/freq=(10,100), eps=0.1
# config 12 is random, phi_max=0.1, step/freq=(5,200), eps=0.01
# config 13 is random, phi_max=0.1, step/freq=(5,200), eps=0.1
# config 14 is random, phi_max=0.1, step/freq=(10,100), eps=0.01
# config 15 is random, phi_max=0.1, step/freq=(10,100), eps=0.1

# TODO: try 0.005, 0.01 for phi_max

# use_dynamics = False # no dynamics noise
# observable_noise = False # no observation noise

# for random in [False, True]: # adversarial, random
#     for phi_max in [0.05, 0.1]:
#         for num_steps, update_freq in zip([5, 10], [200, 100]):
#             for eps in [0.01, 0.1]:
#                 step_size = phi_max / num_steps
#                 phi_array = np.arange(0.0, phi_max + step_size, step_size)

#                 curriculum_config = CurriculumConfig(probability_list=phi_array,
#                                                      update_freq=update_freq,
#                                                      eps=eps,
#                                                      use_max_norm=use_max_norm,
#                                                      use_dynamics=use_dynamics,
#                                                      random=random,
#                                                      observable_noise=observable_noise,
#                                                      num_iter=num_iter)

#                 curriculum_configs.append(curriculum_config)

def get_ddpg_curriculum_configs_dec7():
        curriculum_configs = []
    use_max_norm = False
    num_iter = 2000
    update_freq = 100
    # num_iter = 20
    # update_freq = 1
    phi_max=0.5
    num_steps = 10
    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)

    # config 0 is nominal config
    curriculum_config = CurriculumConfig(adversarial=False,
                                        model_free_adv=False,
                                        probability_list=[0],
                                        update_freq=update_freq,
                                        num_iter=num_iter)

    curriculum_configs.append(curriculum_config)
    
    # configs 1,2,3 changes actual state 
    for eps in [0.01, 0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                             probability_list=phi_array,
                                             update_freq=update_freq,
                                             num_iter=num_iter,
                                             model_free_adv=True,
                                             eps=eps,
                                             use_state=True)
        curriculum_configs.append(curriculum_config)

    # configs 4,5,6 changes action
    for eps in [0.01, 0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                     eps=eps,
                                     probability_list=phi_array,
                                     update_freq=update_freq,
                                     num_iter=num_iter,
                                     model_free_adv=True,
                                     use_action=True)
        curriculum_configs.append(curriculum_config)

    # configs 7,8,9 changes observation
    for eps in [0.01, 0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                     eps=eps,
                                     probability_list=phi_array,
                                     update_freq=update_freq,
                                     num_iter=num_iter,
                                     model_free_adv=True,
                                     use_observation=True)
        curriculum_configs.append(curriculum_config)

    return curriculum_configs



def get_ddpg_curriculum_configs_cartpole():
    curriculum_configs = []
    use_max_norm = False
    num_iter = 500
    update_freq = 50
    # num_iter = 20
    # update_freq = 1
    phi_max=0.8
    num_steps = 8
    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)

    # config 0 is nominal config
    curriculum_config = CurriculumConfig(adversarial=False,
                                        model_free_adv=False,
                                        probability_list=[0],
                                        update_freq=update_freq,
                                        num_iter=num_iter)

    curriculum_configs.append(curriculum_config)
    
    # configs 1,2,3 changes actual state 
    for eps in [0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                             probability_list=phi_array,
                                             update_freq=update_freq,
                                             num_iter=num_iter,
                                             model_free_adv=True,
                                             eps=eps,
                                             use_state=True)
        curriculum_configs.append(curriculum_config)

    # configs 4,5,6 changes action
    for eps in [0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                     eps=eps,
                                     probability_list=phi_array,
                                     update_freq=update_freq,
                                     num_iter=num_iter,
                                     model_free_adv=True,
                                     use_action=True)
        curriculum_configs.append(curriculum_config)

    # configs 7,8,9 changes observation
    for eps in [0.1, 1, 10]:
        curriculum_config = CurriculumConfig(adversarial=False,
                                     eps=eps,
                                     probability_list=phi_array,
                                     update_freq=update_freq,
                                     num_iter=num_iter,
                                     model_free_adv=True,
                                     use_observation=True)
        curriculum_configs.append(curriculum_config)

    return curriculum_configs



### NEW, for dynamics scan
def get_ddpg_curriculum_configs_cartpole_nov28():
    curriculum_configs = []

    # how to do adversarial / random perturbations
    eps = 0.1
    use_max_norm = False

    # number of iterations
    num_iter = 150 # 1000

    # update frequency
    update_freq = 200 # 100

    # config0 is nominal config

    curriculum_config = CurriculumConfig(adversarial=False,
                                         probability_list=[],
                                         update_freq=update_freq,
                                         eps=eps,
                                         use_max_norm=use_max_norm,
                                         use_dynamics=False,
                                         random=False,
                                         observable_noise=False,
                                         num_iter=num_iter)

    curriculum_configs.append(curriculum_config)


    # config1 is random observable
    num_steps = 5
    use_dynamics = False
    observable_noise = True
    random = True
    phi_max = 0.5
    eps = 1.0

    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)

    curriculum_config = CurriculumConfig(adversarial=True,
                                         probability_list=phi_array,
                                         update_freq=update_freq,
                                         eps=eps,
                                         use_max_norm=use_max_norm,
                                         use_dynamics=use_dynamics,
                                         random=random,
                                         observable_noise=observable_noise,
                                         num_iter=num_iter)

    curriculum_configs.append(curriculum_config)

    # config 2, 3, 4 are adversarial, changes observation
    random = False
    for eps in [1.0, 10.0, 0.1]:
        step_size = phi_max / num_steps
        phi_array = np.arange(0.0, phi_max + step_size, step_size)

        curriculum_config = CurriculumConfig(adversarial=True,
                                             probability_list=phi_array,
                                             update_freq=update_freq,
                                             eps=eps,
                                             use_max_norm=use_max_norm,
                                             use_dynamics=use_dynamics,
                                             random=random,
                                             observable_noise=observable_noise,
                                             num_iter=num_iter)

        curriculum_configs.append(curriculum_config)

    # config 5 is random actual state perturbation
    num_steps = 5
    use_dynamics = False
    observable_noise = False
    random = True
    phi_max = 0.5
    eps = 1.0

    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)
    curriculum_config = CurriculumConfig(adversarial=True,
                                         probability_list=phi_array,
                                         update_freq=update_freq,
                                         eps=eps,
                                         use_max_norm=use_max_norm,
                                         use_dynamics=use_dynamics,
                                         random=random,
                                         observable_noise=observable_noise,
                                         num_iter=num_iter)

    curriculum_configs.append(curriculum_config)

    # config 6, 7, 8 is adversarial actual state perturbation
    random = False
    for eps in [1.0, 10.0, 0.1]:
        step_size = phi_max / num_steps
        phi_array = np.arange(0.0, phi_max + step_size, step_size)

        curriculum_config = CurriculumConfig(adversarial=True,
                                             probability_list=phi_array,
                                             update_freq=update_freq,
                                             eps=eps,
                                             use_max_norm=use_max_norm,
                                             use_dynamics=use_dynamics,
                                             random=random,
                                             observable_noise=observable_noise,
                                             num_iter=num_iter)

        curriculum_configs.append(curriculum_config)

    # config 9, 10, 11, 12, 13, 14 are model_free adversarial perturbations
    num_steps = 5
    use_dynamics = False
    observable_noise = False
    random = False
    phi_max = 0.5
    eps = 1.0

    step_size = phi_max / num_steps
    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)
    for bad_action_prob in [0.1, 0.5, 0.9]:
        for bad_action_eps in [1, 10]:
            curriculum_config = CurriculumConfig(adversarial=False,
                                                 probability_list=phi_array,
                                                 update_freq=update_freq,
                                                 eps=eps,
                                                 use_max_norm=use_max_norm,
                                                 use_dynamics=use_dynamics,
                                                 random=random,
                                                 observable_noise=observable_noise,
                                                 num_iter=num_iter,
                                                 model_free_adv=True,
                                                 bad_action_eps=bad_action_eps,
                                                 bad_action_prob=bad_action_prob,)

            curriculum_configs.append(curriculum_config)
    return curriculum_configs

# # config1 is random dynamics (like EPOpt)
# num_steps = 5
# use_dynamics = True
# observable_noise = False
# random = True
# phi_max = 0.5
# eps = 1.0

# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)

# curriculum_config = CurriculumConfig(probability_list=phi_array,
#                                      update_freq=update_freq,
#                                      eps=eps,
#                                      use_max_norm=use_max_norm,
#                                      use_dynamics=use_dynamics,
#                                      random=random,
#                                      observable_noise=observable_noise,
#                                      num_iter=num_iter)

# curriculum_configs.append(curriculum_config)

# # config 2, 3, 4 are adversarial
# random = False
# for eps in [1.0, 10.0, 0.1]:
#     step_size = phi_max / num_steps
#     phi_array = np.arange(0.0, phi_max + step_size, step_size)

#     curriculum_config = CurriculumConfig(probability_list=phi_array,
#                                          update_freq=update_freq,
#                                          eps=eps,
#                                          use_max_norm=use_max_norm,
#                                          use_dynamics=use_dynamics,
#                                          random=random,
#                                          observable_noise=observable_noise,
#                                          num_iter=num_iter)

#     curriculum_configs.append(curriculum_config)




