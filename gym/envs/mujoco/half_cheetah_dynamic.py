import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahDynamicEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        # added
        self.qpos_bounds = [(None, None), (None, None), (None, None), 
                            (-0.52, 1.05), (-0.785, 0.785), (-0.4, 0.785),
                            (-1.0, 0.7), (-1.2, 0.87), (-0.5, 0.5)]
        self.init_params1 = self.model.body_mass.copy() # store initial values
        self.init_params2 = self.model.geom_friction.copy() # store initial values
        self.ratio = 0.5
        self.dynamics_bounds = [((1.0 - self.ratio) * self.init_params1[1], (1.0 + self.ratio) * self.init_params1[1]), 
                                ((1.0 - self.ratio) * self.init_params2[0][0], (1.0 + self.ratio) * self.init_params2[0][0])]

    def _step(self, action):
        xposbefore = self.model.data.qpos[0,0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0,0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run = reward_run, reward_ctrl=reward_ctrl)

    # put current environment parameters into observation 
    def _get_obs(self):
        mass = self.model.body_mass.copy()
        friction = self.model.geom_friction.copy()
        assert(np.unique(friction[:,0]).size == 1)

        return (np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            [mass[1][0], friction[0][0]]
        ])).astype(np.float)

    # added
    def _set_state(self, state):
        # first component of qpos is fixed / unobservable
        # print('qpos')
        # print(self.model.data.qpos)
        # print('qvel')
        # print(self.model.data.qvel)
        qpos = np.concatenate([[self.model.data.qpos.flat[0]], state[:self.model.nq-1]])
        qvel = state[self.model.nq-1:self.model.nq+self.model.nv-1]
        qpos = self.project_state(qpos, self.qpos_bounds)

        # set adversarial dynamics
        dynamics = state[self.model.nq + self.model.nv - 1:]
        dynamics = self.project_dynamics(dynamics, self.dynamics_bounds)
        mass = self.model.body_mass.copy()
        mass[1][0] = dynamics[0]
        friction = self.model.geom_friction.copy()
        # notice we update all friction components here with the same value
        friction[:, 0] = dynamics[1]

        # mass = self.model.body_mass.copy()
        # mass[1] = state[self.model.nq + self.model.nv - 1]
        # friction = self.model.geom_friction.copy()
        # friction[:, 0] = state[self.model.nq + self.model.nv:]

        self.model.body_mass = mass.copy()
        self.model.geom_friction = friction.copy()
        self.set_state(qpos, qvel)
        #print('internal state')
        #print(self._get_obs())

        return self._get_obs()

    # added, return a random set of configuration parameters in the allowed grid
    def get_random_config(self):
        delta1 = self.ratio * self.init_params1[1] 
        delta2 = self.ratio * self.init_params2[0][0] 
        dynamics =  [self.init_params1[1] + np.random.uniform(-delta1, delta1), 
                     self.init_params2[0][0] + np.random.uniform(-delta2, delta2)]
        return self.project_dynamics(dynamics, self.dynamics_bounds)

    # added
    def get_dynamics_bounds(self):
        return self.dynamics_bounds

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        # added, reset config params
        self.model.body_mass = self.init_params1.copy()
        self.model.geom_friction = self.init_params2.copy()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
