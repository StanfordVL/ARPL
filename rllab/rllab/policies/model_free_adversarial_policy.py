import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT
import theano

from rllab.baselines.linear_feature_q import LinearFeatureQ
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

"""
This class implements an adversarial Gaussian MLP Policy that uses curriculum learning.
"""

class ModelFreeAdversarialPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            zero_gradient_cutoff,
            curriculum_list,
            qfunction=None,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            dist_cls=DiagonalGaussian,
            adversarial=True,
            eps=0.1,
            probability=0.0,
            use_dynamics=False,
            random=False,
            observable_noise=False,
            use_max_norm=True,
            record_traj=False,
            set_dynamics=None,
            update_freq=50,
            mask_augmentation=False,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param dist_cls: defines probability distribution over actions

        The following parameters are specific to the CurriculumPolicy Class. Note that the parameters 
        @adversarial, @eps, @probability, @use_dynamics, @random, @observable_noise, and @use_max_norm
        are all included as a formality for initialization, but the initial curriculum configuration will
        overwrite these parameters.

        :param adversarial: whether the policy should incorporate adversarial states during rollout
        :param eps: the strength of the adversarial perturbation
        :param probability: frequency of adversarial updates. If 0, do exactly one update at the beginning of
                            every episode
        :param use_dynamics: if True, generate adversarial dynamics updates, otherwise do adversarial state updates
        :param random: if True, use a random perturbation instead of an adversarial perturbation
        :param observable_noise: if True, don't set adversarial state in the environment, treat it as noise
                                 on observation
        :param zero_gradient_cutoff: determines cutoff index for zero-ing out gradients - this is useful when doing
                                     adversarial dynamics vs. adversarial states, when we only want to compute
                                     gradients for one section of the augmented state vector. We also use this to
                                     determine what the original, non-augmented state size is.
        :param use_max_norm: if True, use Fast Gradient Sign Method (FGSM) to generate adversarial perturbations, else
                             use full gradient ascent
        :param record_traj: if True, rollout dictionaries will contain qpos and qvel trajectories. This is useful for
                            plotting trajectories.
        :param set_dynamics: if provided, the next rollout initializes the environment to the passed dynamics.
        :param curriculum_list: list of configurations, in order, to add to curriculum
        :param update_freq: update the curriculum every @update_freq iters
        :param mask_augmentation: if True, don't augment the state (even though the environment augments the state with 
                                  the dynamics parameters, the policy will ignore these dimensions)

        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # TODO: make a more elegant solution to this
        # This is here because we assume the original, unaugmented state size is provided.
        assert(zero_gradient_cutoff is not None)

        # if we're ignoring state augmentation, modify observation size / network size accordingly
        if mask_augmentation:
            obs_dim = zero_gradient_cutoff

        # create network
        if mean_network is None:
            mean_network = MLP(
                input_shape=(obs_dim,),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )
        self._mean_network = mean_network

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                std_network = MLP(
                    input_shape=(obs_dim,),
                    input_layer=mean_network.input_layer,
                    output_dim=action_dim,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=std_hidden_nonlinearity,
                    output_nonlinearity=None,
                )
                l_log_std = std_network.output_layer
            else:
                l_log_std = ParamLayer(
                    mean_network.input_layer,
                    num_units=action_dim,
                    param=lasagne.init.Constant(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )

        self.min_std = min_std

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(min_std))

        self._mean_var, self._log_std_var = mean_var, log_std_var

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        # take exponential for the actual standard dev
        self._tru_std_var = TT.exp(self._log_std_var)

        # take gradients of mean network, exponential of std network wrt L2 norm
        self._mean_grad = theano.grad(self._mean_var.norm(2), obs_var)
        self._std_grad = theano.grad(self._tru_std_var.norm(2), obs_var, disconnected_inputs='warn')

        self._dist = dist_cls(action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(ModelFreeAdversarialPolicy, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

        # function to get gradients
        self._f_grad_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[self._mean_grad, self._std_grad]
        )

        # initialize adversarial parameters
        self.adversarial = adversarial
        self.eps = eps
        self.probability = probability
        self.use_dynamics = use_dynamics
        self.random = random
        self.observable_noise = observable_noise
        self.use_max_norm = use_max_norm
        self.zero_gradient_cutoff = zero_gradient_cutoff
        self.record_traj = record_traj
        self.set_dynamics = set_dynamics
        self.mask_augmentation = mask_augmentation

        self.curriculum_list = list(curriculum_list)
        self.update_freq = update_freq

        # initialize q function
        self.qfunction = qfunction
        if self.qfunction is None:
            self.qfunction = LinearFeatureQ(env_spec=env_spec)

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)

        # ignore augmented part of state if necessary
        if self.mask_augmentation:
            flat_obs = flat_obs[:self.zero_gradient_cutoff]

        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)

        # ignore augmented part of state if necessary
        if self.mask_augmentation:
            flat_obs = flat_obs[:, :self.zero_gradient_cutoff]

        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (TT.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * TT.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist


    ### Class methods for generating adversarial states


    def get_adv_gradient(self, observation):
        """

        This method is specifically for adversarial training of the Gaussian MLP Policy.
        We use the loss ||mu||^2 + ||sigma||^2 and compute the gradient of the input with
        respect to the loss.

        :param observation: an observation vector to compute the gradient with respect to
        :return: gradient of loss with respect to observation
        """
        flat_obs = self.observation_space.flatten(observation)

        # ignore augmented part of state if necessary
        if self.mask_augmentation:
            flat_obs = flat_obs[:self.zero_gradient_cutoff]

        mean_grad, std_grad = [x[0] for x in self._f_grad_dist([flat_obs])]

        # zero-out some components if augmenting state
        if not self.mask_augmentation:
            if self.use_dynamics:
                mean_grad[:self.zero_gradient_cutoff] = 0.0
                std_grad[:self.zero_gradient_cutoff] = 0.0
            else:
                mean_grad[self.zero_gradient_cutoff:] = 0.0
                std_grad[self.zero_gradient_cutoff:] = 0.0

        return mean_grad + std_grad


    def set_adversarial_state(self, env, full_state, observation):
        """

        This method is used to set the state part of an augmented state vector. The first @zero_gradient_cutoff
        components correspond to the observations while the last components correspond to the dynamics
        parameters. It also sets the environment dynamics as well.

        :param env: the environment
        :param full_state: the full augmented state vector
        :param observation: the observation to set
        :return: the modified full state vector
        """
        full_state[:self.zero_gradient_cutoff] = observation
        full_state[:self.zero_gradient_cutoff] = env._set_state(full_state)[:self.zero_gradient_cutoff]
        return full_state  # technically not necessary

    def set_adversarial_dynamics(self, env, full_state, dynamics, scale=None):
        """

        This method is used to set the state part of an augmented state vector. The first @zero_gradient_cutoff
        components correspond to the observations while the last components correspond to the dynamics
        parameters. It also sets the environment dynamics as well.

        :param env: the environment
        :param full_state: the full augmented state vector
        :param dynamics: the dynamics parameters to set
        :param scale: if not None, scale the dynamics by this factor before updating the state vector
        :return: the modified full state vector
        """
        if scale is None:
            scale = 1.0

        if self.mask_augmentation:
            # just use dynamics part of full_state vector to set environment dynamics
            new_state = full_state.copy()
            new_state[self.zero_gradient_cutoff:] = dynamics
            env._set_state(new_state)
        else:
            full_state[self.zero_gradient_cutoff:] = dynamics
            full_state[self.zero_gradient_cutoff:] = env._set_state(full_state)[self.zero_gradient_cutoff:]
            full_state[self.zero_gradient_cutoff:] *= scale 
        return full_state  # technically not necessary


    def do_perturbation(self, env, state, is_start=False, scale=None):
        """

        This method computes an adversarial / random perturbation with respect to the observation (if necessary)
        and applies it. It also sets the internal environment state appropriately (if necessary).

        :param env: the environment
        :param state: the observation (full state vector)
        :param is_start: if True, this is the beginning of an episode. This guarantees an update for when the
                             agent's probability is 0.0.
        :param scale: if not None, scale the dynamics by this factor before updating the state vector. 
        :return: perturbed state
        """

        # If @adversarial is not set, this is a no-op
        if not self.adversarial:
            return state

        # If @probability is 0 and it isn't the beginning of the episode, this is a no-op
        if self.probability == 0.0 and not is_start:
            return state

        # If @probability > 0 and we lose the coin flip this is a no-op
        if self.probability > 0 and np.random.binomial(1, 1.0 - self.probability):
            return state

        # Adversarial dynamics generation
        if self.use_dynamics:
            # Sample uniformly from dynamics grid first.
            dynamics = env.get_random_config()
            state = self.set_adversarial_dynamics(env=env, full_state=state, dynamics=dynamics, scale=scale)

            # Do an adversarial perturbation on top of the random dynamics.
            # Note that we compute the perturbation with respect to the new random dynamics.
            if not self.random:
                # state must be augmented for adversarial dynamics perturbations
                assert(not self.mask_augmentation) 

                # gradient of loss wrt new sampled dynamics
                grad = self.get_adv_gradient(state)

                if self.use_max_norm:
                    state += self.eps * np.sign(grad)  # FGSM
                else:
                    state += self.eps * grad  # gradient ascent

        # Adversarial state generation
        else:
            # gradient of loss wrt state
            grad = self.get_adv_gradient(state)

            if self.random:
                # For random generation, sample uniformly, with bounds given by gradient.
                grad = np.abs(grad)
                grad = np.random.uniform(-grad, grad)
            if self.use_max_norm:
                # note: use len(grad) to handle both augmented and non-augmented gradients, since
                # the grad size depends on @self.mask_augmentation
                state[:len(grad)] += self.eps * np.sign(grad)  # FGSM
            else:
                state[:len(grad)] += self.eps * grad  # gradient ascent

        # only change environment if no observation noise
        if not self.observable_noise:
            if scale is not None:
                # scale down to set environment dynamics correctly
                env_state = state.copy()
                env_state[self.zero_gradient_cutoff:] /= scale
                env_state = env._set_state(env_state)
                state = env_state.copy()
                state[self.zero_gradient_cutoff:] *= scale
            else:
                state = env._set_state(state)

        return state

    def set_config(self, config):
        """
        Note, only sets policy config parameters, not the algo ones.

        :param config:
        :return:
        """
        self.adversarial = config.adversarial
        self.eps = config.eps
        self.probability = config.probability
        self.use_dynamics = config.use_dynamics
        self.random = config.random
        self.observable_noise = config.observable_noise
        self.use_max_norm = config.use_max_norm

    def sample_from_curriculum(self, curriculum):
        """
        Uniformly samples from curriculum and sets the internal policy parameters accordingly.
        :return: the config that was set
        """
        sample_config = np.random.choice(curriculum)
        self.set_config(sample_config)
        return sample_config

    def set_curriculum_config(self, curriculum_config):
        """
        :param curriculum_config: CurriculumConfig object
        :return:
        """
        self.curriculum_list = curriculum_config.curriculum_list
        self.update_freq = curriculum_config.update_freq

