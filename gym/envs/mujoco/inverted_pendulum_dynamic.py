import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumDynamicEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

        # added
        self.qpos_bounds = [(-1.0, 1.0), (-90.0, 90.0)]
        self.init_params = self.model.body_mass.copy() # store initial values
        self.ratio = 0.5
        self.dynamics_bounds = [((1.0 - self.ratio) * self.init_params[1], (1.0 + self.ratio) * self.init_params[1]), 
                                ((1.0 - self.ratio) * self.init_params[2], (1.0 + self.ratio) * self.init_params[2])]

    def _step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)

        # added, reset config params
        #print('model config before reset: {}'.format(self.model.body_mass))
        self.model.body_mass = self.init_params.copy()

        #print('init_state: {}'.format(self._get_obs()))
        return self._get_obs()

    # added (changed method)
    def _get_obs(self):
        # put current environment parameters into observation 
        mass = self.model.body_mass.copy()
        return np.concatenate([self.model.data.qpos, self.model.data.qvel, [mass[1], mass[2]]]).ravel()

    # added
    def _set_state(self, state):
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:self.model.nq+self.model.nv]
        qpos = self.project_state(qpos, self.qpos_bounds)

        # set adversarial dynamics
        dynamics = state[self.model.nq + self.model.nv:]
        dynamics = self.project_dynamics(dynamics, self.dynamics_bounds)
        mass = self.model.body_mass.copy()
        mass[1] = dynamics[0]
        mass[2] = dynamics[1]

        # set adversarial dynamics
        # mass = self.model.body_mass.copy()
        # mass[1] = state[self.model.nq + self.model.nv]
        # mass[2] = state[self.model.nq + self.model.nv + 1]
        #print('true config before: {}'.format(self.model.body_mass))
        #print('mass: {}'.format(mass))
        self.model.body_mass = mass.copy()
        #print('true config after: {}'.format(self.model.body_mass))

        self.set_state(qpos, qvel)

        return self._get_obs()

    # added, return a random set of configuration parameters in the allowed grid
    def get_random_config(self):
        delta1 = self.ratio * self.init_params[1] 
        delta2 = self.ratio * self.init_params[2] 
        dynamics =  [self.init_params[1] + np.random.uniform(-delta1, delta1), 
                     self.init_params[2] + np.random.uniform(-delta2, delta2)]
        return self.project_dynamics(dynamics, self.dynamics_bounds)

    # added
    def get_dynamics_bounds(self):
        return self.dynamics_bounds

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
