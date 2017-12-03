"""

Use these configurations to use curriculum learning.

This module provides configurations for running experiments.
It also makes it easy to define new configurations for experiments
that we might want to run.

"""

import numpy as np
from config import Config

class CurriculumConfig(object):
    def __init__(self, adversarial, probability_list, update_freq, 
        eps, use_max_norm, use_dynamics, 
        random, observable_noise, num_iter,
        model_free_adv=False, bad_action_eps=0.1, bad_action_prob=0.3, model_free_max_norm=False):
        """

        :param probability_list: list of probabilities to make configs for
        :param update_freq: update frequency for curriculum
        :param eps: strength of adversarial peturbation
        :param use_max_norm: if True, use FGSM, else, use gradient ascent
        """
        self.probability_list = probability_list
        self.update_freq = update_freq
        self.curriculum_list = []

        # nominal
        config0 = Config(adversarial=False,
                         eps=eps,
                         probability=-1.0,
                         use_dynamics=False,
                         random=False,
                         observable_noise=False,
                         use_max_norm=use_max_norm,
                         num_iter=num_iter,
                         model_free_adv=False,
                         bad_action_eps=bad_action_eps,
                         bad_action_prob=bad_action_prob)

        self.curriculum_list.append(config0)

        for phi in probability_list:
            phi_config = Config(adversarial=adversarial,
                                eps=eps,
                                probability=phi,
                                use_dynamics=use_dynamics,
                                random=random,
                                observable_noise=observable_noise,
                                use_max_norm=use_max_norm,
                                num_iter=num_iter,
                                model_free_adv=model_free_adv,
                                bad_action_eps=bad_action_eps,
                                bad_action_prob=bad_action_prob)
            self.curriculum_list.append(phi_config)

    def set_batch_size(self, batch_size):
        for config in self.curriculum_list:
            config.batch_size = batch_size

    def print(self):
        print('=======================')
        counter = 0
        for one_config in self.curriculum_list:
            print('Config {}:'.format(counter))
            one_config.print_params()
            counter += 1
        print('=======================')


curriculum_configs = []

# how to do adversarial / random perturbations
eps = 0.1
use_max_norm = False

# number of iterations
num_iter = 2000 # 1000

# update frequency
update_freq = 200 # 100

# config0 is nominal config

# curriculum_config = CurriculumConfig(probability_list=[],
#                                      update_freq=update_freq,
#                                      eps=eps,
#                                      use_max_norm=use_max_norm,
#                                      use_dynamics=False,
#                                      random=False,
#                                      observable_noise=False,
#                                      num_iter=num_iter)

# curriculum_configs.append(curriculum_config)

# # number of steps between 0 and phi_max
# num_steps = 5 # 10

# # config 1 is adversarial process noise
# # config 2 is random process noise
# # config 3 is adversarial dynamics noise
# # config 4 is random dynamics noise

# for use_dynamics in [False, True]: # process noise, dynamics noise
#     observable_noise = False # no observation noise
#     for random in [False, True]: # adversarial, random
#         for phi_max in [0.5]:
#             step_size = phi_max / num_steps
#             phi_array = np.arange(0.0, phi_max + step_size, step_size)

#             curriculum_config = CurriculumConfig(probability_list=phi_array,
#                                                  update_freq=update_freq,
#                                                  eps=eps,
#                                                  use_max_norm=use_max_norm,
#                                                  use_dynamics=use_dynamics,
#                                                  random=random,
#                                                  observable_noise=observable_noise,
#                                                  num_iter=num_iter)

#             curriculum_configs.append(curriculum_config)

#
# # configs 10-14 and 15-19 have double update_freq, half steps
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# # number of steps between 0 and phi_max
# num_steps = 5
#
# for random in [True, False]:
#     for phi_max in [0.1, 0.2, 0.3, 0.4, 0.5]:
#         step_size = phi_max / num_steps
#         phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
#         curriculum_config = CurriculumConfig(probability_list=phi_array,
#                                              update_freq=update_freq,
#                                              eps=eps,
#                                              use_max_norm=use_max_norm,
#                                              use_dynamics=use_dynamics,
#                                              random=random,
#                                              observable_noise=observable_noise,
#                                              num_iter=num_iter)
#
#         curriculum_configs.append(curriculum_config)

# curriculum_configs = []
#
# eps = 0.1
# use_max_norm = False
#
# # control type of perturbation
# use_dynamics = True
# observable_noise = False
#
# # random or adversarial
# random = True
#
# ### Config 0
#
# # phis to scan over
# phi_max = 0.1
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config0 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config0)
#
# ### Config 1
#
# # phis to scan over
# phi_max = 0.2
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config1 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config1)
#
# ### Config 2
#
# # phis to scan over
# phi_max = 0.3
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config2 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config2)
#
# ### Config 3
#
# # phis to scan over
# phi_max = 0.4
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config3 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config3)
#
# ### Config 4
#
# # phis to scan over
# phi_max = 0.5
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config4 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config4)
#
#
# eps = 0.1
# use_max_norm = False
#
# # control type of perturbation
# use_dynamics = True
# observable_noise = False
#
# # random or adversarial
# random = False
#
# ### Config 5
#
# # phis to scan over
# phi_max = 0.1
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config5 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config5)
#
# ### Config 6
#
# # phis to scan over
# phi_max = 0.2
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config6 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config6)
#
# ### Config 7
#
# # phis to scan over
# phi_max = 0.3
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config7 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config7)
#
# ### Config 8
#
# # phis to scan over
# phi_max = 0.4
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config8 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config8)
#
# ### Config 9
#
# # phis to scan over
# phi_max = 0.5
# num_steps = 10
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 50
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config9 = CurriculumConfig(probability_list=phi_array,
#                                       update_freq=update_freq,
#                                       eps=eps,
#                                       use_max_norm=use_max_norm,
#                                       use_dynamics=use_dynamics,
#                                       random=random,
#                                       observable_noise=observable_noise,
#                                       num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config9)
#
#
# eps = 0.1
# use_max_norm = False
#
# # control type of perturbation
# use_dynamics = True
# observable_noise = False
#
# # random or adversarial
# random = True
#
# ### Config 10
#
# # phis to scan over
# phi_max = 0.1
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config10 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config10)
#
# ### Config 11
#
# # phis to scan over
# phi_max = 0.2
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config11 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config11)
#
# ### Config 12
#
# # phis to scan over
# phi_max = 0.3
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config12 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config12)
#
# ### Config 13
#
# # phis to scan over
# phi_max = 0.4
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config13 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config13)
#
# ### Config 14
#
# # phis to scan over
# phi_max = 0.5
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config14 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config14)
#
#
# eps = 0.1
# use_max_norm = False
#
# # control type of perturbation
# use_dynamics = True
# observable_noise = False
#
# # random or adversarial
# random = False
#
# ### Config 15
#
# # phis to scan over
# phi_max = 0.1
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config15 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config15)
#
# ### Config 16
#
# # phis to scan over
# phi_max = 0.2
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config16 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config16)
#
# ### Config 17
#
# # phis to scan over
# phi_max = 0.3
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config17 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config17)
#
# ### Config 18
#
# # phis to scan over
# phi_max = 0.4
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config18 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config18)
#
# ### Config 19
#
# # phis to scan over
# phi_max = 0.5
# num_steps = 5
# step_size = phi_max / num_steps
# phi_array = np.arange(0.0, phi_max + step_size, step_size)
#
# # update frequency
# update_freq = 100
#
# # number of iterations
# num_iter = 1000
#
# curriculum_config19 = CurriculumConfig(probability_list=phi_array,
#                                        update_freq=update_freq,
#                                        eps=eps,
#                                        use_max_norm=use_max_norm,
#                                        use_dynamics=use_dynamics,
#                                        random=random,
#                                        observable_noise=observable_noise,
#                                        num_iter=num_iter)
#
# curriculum_configs.append(curriculum_config19)

