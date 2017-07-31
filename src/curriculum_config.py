"""

Use these configurations to use curriculum learning.

This file contains configurations for experiments with process noise.

"""

import numpy as np
from config import Config
from curriculum_config import CurriculumConfig

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

use_dynamics = False # no dynamics noise
observable_noise = False # no observation noise

for random in [False, True]: # adversarial, random
    for phi_max in [0.05, 0.1]:
        for num_steps, update_freq in zip([5, 10], [200, 100]):
            for eps in [0.01, 0.1]:
                step_size = phi_max / num_steps
                phi_array = np.arange(0.0, phi_max + step_size, step_size)

                curriculum_config = CurriculumConfig(probability_list=phi_array,
                                                     update_freq=update_freq,
                                                     eps=eps,
                                                     use_max_norm=use_max_norm,
                                                     use_dynamics=use_dynamics,
                                                     random=random,
                                                     observable_noise=observable_noise,
                                                     num_iter=num_iter)

                curriculum_configs.append(curriculum_config)


### NEW, for dynamics scan

curriculum_configs = []

# how to do adversarial / random perturbations
eps = 0.1
use_max_norm = False

# number of iterations
num_iter = 2000 # 1000

# update frequency
update_freq = 200 # 100

# config0 is nominal config

curriculum_config = CurriculumConfig(probability_list=[],
                                     update_freq=update_freq,
                                     eps=eps,
                                     use_max_norm=use_max_norm,
                                     use_dynamics=False,
                                     random=False,
                                     observable_noise=False,
                                     num_iter=num_iter)

curriculum_configs.append(curriculum_config)

# config1 is random dynamics (like EPOpt)
num_steps = 5
use_dynamics = True
observable_noise = False
random = True
phi_max = 0.5
eps = 1.0

step_size = phi_max / num_steps
phi_array = np.arange(0.0, phi_max + step_size, step_size)

curriculum_config = CurriculumConfig(probability_list=phi_array,
                                     update_freq=update_freq,
                                     eps=eps,
                                     use_max_norm=use_max_norm,
                                     use_dynamics=use_dynamics,
                                     random=random,
                                     observable_noise=observable_noise,
                                     num_iter=num_iter)

curriculum_configs.append(curriculum_config)

# config 2, 3, 4 are adversarial
random = False
for eps in [1.0, 10.0, 0.1]:
    step_size = phi_max / num_steps
    phi_array = np.arange(0.0, phi_max + step_size, step_size)

    curriculum_config = CurriculumConfig(probability_list=phi_array,
                                         update_freq=update_freq,
                                         eps=eps,
                                         use_max_norm=use_max_norm,
                                         use_dynamics=use_dynamics,
                                         random=random,
                                         observable_noise=observable_noise,
                                         num_iter=num_iter)

    curriculum_configs.append(curriculum_config)




