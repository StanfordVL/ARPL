"""

This module provides configurations for running experiments.
It also makes it easy to define new configurations for experiments
that we might want to run.

"""

eps = 0.1
use_max_norm = False

class Config(object):
    def __init__(self, adversarial, eps, probability, use_dynamics, random, observable_noise, use_max_norm,
                 num_iter=500, batch_size=25000, discount=0.995, hidden_sizes=(64, 64), adaptive_std=False,
                 step_size=0.01, gae_lambda=0.97, num_workers=1, plot_learning_curve=True, mask_augmentation=False):
        """

        :param adversarial: if True, use adversarial states
        :param eps: strength of adversarial peturbation
        :param probability: frequency of adversarial perturbation
        :param use_dynamics: if True, do adversarial dynamics, else adversarial states
        :param random: if True, do random perturbation instead of adversarial
        :param observable_noise: if True, don't update internal environment state with perturbed state
        :param use_max_norm: if True, use FGSM, else, use gradient ascent
        :param num_iter: number of training iterations
        :param batch_size: number of total samples to collect over all trajectories per iteration
        :param discount: discount factor
        :param hidden_sizes: architecture of policy network
        :param adaptive_std: if True, adaptively learn std
        :param step_size: step size of learning
        :param gae_lambda: TRPO parameter
        :param num_workers: number of parallel workers for sampling trajectories
        :param plot_learning_curve: if True, collect reward curve during training
        :param mask_augmentation: if True, don't augment the state (even though the environment augments the state with 
                                  the dynamics parameters, the policy will ignore these dimensions)

        """

        self.adversarial = adversarial
        self.eps = eps
        self.probability = probability
        self.use_dynamics = use_dynamics
        self.random = random
        self.observable_noise = observable_noise
        self.use_max_norm = use_max_norm
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.discount = discount
        self.hidden_sizes = hidden_sizes
        self.adaptive_std = adaptive_std
        self.step_size = step_size
        self.gae_lambda = gae_lambda
        self.num_workers = num_workers
        self.plot_learning_curve = plot_learning_curve
        self.mask_augmentation = mask_augmentation

    def print_params(self):
        print("adversarial : {}".format(self.adversarial))
        print("eps : {}".format(self.eps))
        print("probability : {}".format(self.probability))
        print("use_dynamics : {}".format(self.use_dynamics))
        print("random : {}".format(self.random))
        print("observable_noise : {}".format(self.observable_noise))
        print("use_max_norm : {}".format(self.use_max_norm))
        print("num_iter : {}".format(self.num_iter))
        print("batch_size : {}".format(self.batch_size))
        print("discount : {}".format(self.discount))
        print("hidden_sizes : {}".format(self.hidden_sizes))
        print("adaptive_std : {}".format(self.adaptive_std))
        print("step_size : {}".format(self.step_size))
        print("gae_lambda : {}".format(self.gae_lambda))
        print("num_workers : {}".format(self.num_workers))
        print("plot_learning_curve : {}".format(self.plot_learning_curve))
        print("mask_augmentation : {}".format(self.mask_augmentation))

configs = []

# baseline model (no adversarial training)
config0 = Config(adversarial=False,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=False,
                 random=False,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config0)

# dynamics noise, random, phi=0.0
config1 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=True,
                 random=True,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config1)

# dynamics noise, adversarial, phi=0.0
config2 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=True,
                 random=False,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config2)

# process noise, random, phi=0.0
config3 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=False,
                 random=True,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config3)

# process noise, adversarial, phi=0.0
config4 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=False,
                 random=False,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config4)

# observation noise, random, phi=0.0
config5 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=False,
                 random=True,
                 observable_noise=True,
                 use_max_norm=use_max_norm)

configs.append(config5)

# observation noise, adversarial, phi=0.0
config6 = Config(adversarial=True,
                 eps=eps,
                 probability=0.0,
                 use_dynamics=False,
                 random=False,
                 observable_noise=True,
                 use_max_norm=use_max_norm)

configs.append(config6)

# dynamics noise, random, phi=0.1
config7 = Config(adversarial=True,
                 eps=eps,
                 probability=0.1,
                 use_dynamics=True,
                 random=True,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config7)

# dynamics noise, adversarial, phi=0.1
config8 = Config(adversarial=True,
                 eps=eps,
                 probability=0.1,
                 use_dynamics=True,
                 random=False,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config8)

# process noise, random, phi=0.1
config9 = Config(adversarial=True,
                 eps=eps,
                 probability=0.1,
                 use_dynamics=False,
                 random=True,
                 observable_noise=False,
                 use_max_norm=use_max_norm)

configs.append(config9)

# process noise, adversarial, phi=0.1
config10 = Config(adversarial=True,
                  eps=eps,
                  probability=0.1,
                  use_dynamics=False,
                  random=False,
                  observable_noise=False,
                  use_max_norm=use_max_norm)

configs.append(config10)

# observation noise, random, phi=0.1
config11 = Config(adversarial=True,
                  eps=eps,
                  probability=0.1,
                  use_dynamics=False,
                  random=True,
                  observable_noise=True,
                  use_max_norm=use_max_norm)

configs.append(config11)

# observation noise, adversarial, phi=0.1
config12 = Config(adversarial=True,
                  eps=eps,
                  probability=0.1,
                  use_dynamics=False,
                  random=False,
                  observable_noise=True,
                  use_max_norm=use_max_norm)

configs.append(config12)

# dynamics noise, random, phi=0.5
config13 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=True,
                  random=True,
                  observable_noise=False,
                  use_max_norm=use_max_norm)

configs.append(config13)

# dynamics noise, adversarial, phi=0.5
config14 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=True,
                  random=False,
                  observable_noise=False,
                  use_max_norm=use_max_norm)

configs.append(config14)

# process noise, random, phi=0.5
config15 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=False,
                  random=True,
                  observable_noise=False,
                  use_max_norm=use_max_norm)

configs.append(config15)

# process noise, adversarial, phi=0.5
config16 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=False,
                  random=False,
                  observable_noise=False,
                  use_max_norm=use_max_norm)

configs.append(config16)

# observation noise, random, phi=0.5
config17 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=False,
                  random=True,
                  observable_noise=True,
                  use_max_norm=use_max_norm)

configs.append(config17)

# observation noise, adversarial, phi=0.5
config18 = Config(adversarial=True,
                  eps=eps,
                  probability=0.5,
                  use_dynamics=False,
                  random=False,
                  observable_noise=True,
                  use_max_norm=use_max_norm)

configs.append(config18)

