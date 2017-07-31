"""

Use these configurations to test on several different phi values.

This module provides configurations for running experiments.
It also makes it easy to define new configurations for experiments
that we might want to run.

"""

import numpy as np
from config import Config

use_max_norm = False

# phis to scan over
phi_max = 0.5
num_steps = 20
step_size = phi_max / num_steps
phi_array = np.arange(0.0, phi_max + step_size, step_size)


def get_phi_configs(use_dynamics, observable_noise, random, eps):
    phi_configs = []

    # baseline model (no adversarial training)
    config0 = Config(adversarial=False,
                     eps=eps,
                     probability=-1.0,
                     use_dynamics=False,
                     random=False,
                     observable_noise=False,
                     use_max_norm=use_max_norm)

    phi_configs.append(config0)

    for phi in phi_array:
        phi_config = Config(adversarial=True,
                            eps=eps,
                            probability=phi,
                            use_dynamics=use_dynamics,
                            random=random,
                            observable_noise=observable_noise,
                            use_max_norm=use_max_norm)
        phi_configs.append(phi_config)

    return phi_configs

# used to be eps = 0.1, and eps = [0.1, 0.01]

all_phi_configs = []
for eps in [0.1, 0.01, 1.0, 10.0]:
    for use_dynamics in [False, True]: # process noise, dynamics noise
        observable_noise = False # no observation noise
        for random in [False, True]: # adversarial, random
            phi_configs = get_phi_configs(use_dynamics=use_dynamics, observable_noise=observable_noise, random=random, eps=eps)
            all_phi_configs.append(phi_configs)

