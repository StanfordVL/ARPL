import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import batch_norm
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.policies.base import Policy
import theano.tensor as TT
import theano
import numpy as np
# Not done yet.
class DeterministicMLPCurriculumPolicy(Policy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            zero_gradient_cutoff,
            curriculum_list,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.rectify,
            hidden_W_init=LI.HeUniform(),
            hidden_b_init=LI.Constant(0.),
            output_nonlinearity=NL.tanh,
            output_W_init=LI.Uniform(-3e-3, 3e-3),
            output_b_init=LI.Uniform(-3e-3, 3e-3),
            bn=False,
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
            qf=None,
            model_free_eps=0.5,
            model_free_phi=0.5,
            model_free_adv_observation=False,
            model_free_adv_action=False,
            model_free_adv_state=False,
            model_free_max_norm=False,
            ):

        Serializable.quick_init(self, locals())

        l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim))

        l_hidden = l_obs
        if bn:
            l_hidden = batch_norm(l_hidden)

        for idx, size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=size,
                W=hidden_W_init,
                b=hidden_b_init,
                nonlinearity=hidden_nonlinearity,
                name="h%d" % idx
            )
            if bn:
                l_hidden = batch_norm(l_hidden)

        l_output = L.DenseLayer(
            l_hidden,
            num_units=env_spec.action_space.flat_dim,
            W=output_W_init,
            b=output_b_init,
            nonlinearity=output_nonlinearity,
            name="output"
        )

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers

        action_var = L.get_output(l_output, deterministic=True)
        self._output_layer = l_output

        self._f_actions = ext.compile_function([l_obs.input_var], action_var)

        self._mean_grad = theano.grad(action_var.norm(2), l_obs.input_var)
        self._f_grad = ext.compile_function(
            inputs=[l_obs.input_var],
            outputs=[self._mean_grad]
        )

        super(DeterministicMLPCurriculumPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [l_output])

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
        # NEW, use qfunction to do model free adversarial perturbation
        self.qf = qf
        self.model_free_adv_observation = model_free_adv_observation
        self.model_free_adv_action = model_free_adv_action
        self.model_free_adv_state = model_free_adv_state
        self.model_free_max_norm = model_free_max_norm

        self.curriculum_list = list(curriculum_list)
        self.update_freq = update_freq

    def get_action(self, observation):
        if len(observation.shape) == 1:
            observation = observation.reshape(1,-1)
        if self.model_free_adv_action:
            if np.random.uniform() < self.model_free_phi:
                bad_action = self.get_bad_action(observation)
                return bad_action, dict()
        action = self._f_actions(observation)[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        return L.get_output(self._output_layer, obs_var)

    ### Class methods for generating adversarial states

    def get_bad_action(self, observation):
        if len(observation.shape) == 1:
            observation = observation.reshape(1,-1)
        if not self.model_free_max_norm:
            action = self._f_actions(observation)[0]
            gradient = self.qf._f_qgrad_action(observation.reshape([1, -1]), action.reshape([1, -1]))[0]
            return action + self.model_free_eps * gradient
        else:
            action = self._f_actions(observation)[0]
            gradient = self.qf._f_qgrad_action(observation.reshape([1, -1]), action.reshape([1, -1]))[0]
            return action + self.model_free_eps * gradient / np.abs(gradient)
    
    def get_bad_state(self, observation):
        if len(observation.shape) == 1:
            observation = observation.reshape(1,-1)
        if not self.model_free_max_norm:
            action = self._f_actions(observation)[0]
            gradient = self.qf._f_qgrad_state(observation.reshape([1, -1]), action.reshape([1, -1]))
            return (observation + self.model_free_eps * gradient)[0]
        else:
            action = self._f_actions(observation)[0]
            gradient = self.qf._f_qgrad_state(observation.reshape([1, -1]), action.reshape([1, -1]))
            return (observation + self.model_free_eps * gradient / np.abs(gradient))[0]

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

        mean_grad = [x[0] for x in self._f_grad([flat_obs])][0]
        # zero-out some components if augmenting state
        if not self.mask_augmentation:
            if self.use_dynamics:
                mean_grad[:self.zero_gradient_cutoff] = 0.0
                # std_grad[:self.zero_gradient_cutoff] = 0.0
            else:
                mean_grad[self.zero_gradient_cutoff:] = 0.0
                # std_grad[self.zero_gradient_cutoff:] = 0.0

        return mean_grad


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
        # Q function based adversarial
        if self.model_free_adv_state or self.model_free_adv_observation:
            if np.random.uniform() < self.model_free_phi:
                state = self.get_bad_state(state)
                if self.model_free_adv_state:
                    if scale is not None:
                        # scale down to set environment dynamics correctly
                        env_state = state.copy()
                        env_state[self.zero_gradient_cutoff:] /= scale
                        env_state = env._set_state(env_state)
                        state = env_state.copy()
                        state[self.zero_gradient_cutoff:] *= scale
                    else:
                        state = env._set_state(state)

        # If @adversarial is not set, this is a no-op
        if self.adversarial:   
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

        self.model_free_eps = config.model_free_eps
        self.model_free_phi = config.model_free_phi
        self.model_free_adv_observation = config.model_free_adv_observation
        self.model_free_adv_action = config.model_free_adv_action
        self.model_free_adv_state = config.model_free_adv_state
        self.model_free_max_norm = config.model_free_max_norm


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




