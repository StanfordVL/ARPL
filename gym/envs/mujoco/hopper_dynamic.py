import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperDynamicEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        # added
        self.qpos_bounds = [(None, None), (None, None), (None, None), 
                            (-150.0, 0.0), (-150.0, 0.0), (-45.0, 45.0)]
        self.init_params1 = self.model.body_mass.copy() # store initial values
        self.init_params2 = self.model.geom_friction.copy() # store initial values
        self.ratio = 0.5
        self.dynamics_bounds = [((1.0 - self.ratio) * self.init_params1[1], (1.0 + self.ratio) * self.init_params1[1]), 
                                ((1.0 - self.ratio) * self.init_params2[4][0], (1.0 + self.ratio) * self.init_params2[4][0])]

    def _step(self, a):
        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    # put current environment parameters into observation 
    def _get_obs(self):
        mass = self.model.body_mass.copy()
        friction = self.model.geom_friction.copy()

        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10),
            [mass[1], friction[4][0]]
        ]).astype(float)

    # added
    def _set_state(self, state):
        # first component of qpos is fixed / unobservable
        # print('qpos')
        # print(self.model.data.qpos)
        # print('qvel')
        # print(self.model.data.qvel)
        qpos = np.concatenate([[self.model.data.qpos.flat[0]], state[:self.model.nq-1]])
        qvel = np.clip(state[self.model.nq-1:self.model.nq+self.model.nv-1],-10,10)
        qpos = self.project_state(qpos, self.qpos_bounds)

        # set adversarial dynamics
        dynamics = state[self.model.nq + self.model.nv - 1:]
        dynamics = self.project_dynamics(dynamics, self.dynamics_bounds)
        mass = self.model.body_mass.copy()
        mass[1] = dynamics[0]
        friction = self.model.geom_friction.copy()
        friction[4][0] = dynamics[1]

        # set adversarial dynamics
        # mass = self.model.body_mass.copy()
        # mass[1] = state[self.model.nq + self.model.nv - 1]
        # friction = self.model.geom_friction.copy()
        # friction[4][0] = state[-1]
        self.model.body_mass = mass.copy()
        self.model.geom_friction = friction.copy()

        self.set_state(qpos, qvel)

        return self._get_obs()

    # added, return a random set of configuration parameters in the allowed grid
    def get_random_config(self):
        delta1 = self.ratio * self.init_params1[1] 
        delta2 = self.ratio * self.init_params2[4][0] 
        dynamics =  [self.init_params1[1] + np.random.uniform(-delta1, delta1), 
                     self.init_params2[4][0] + np.random.uniform(-delta2, delta2)]
        return self.project_dynamics(dynamics, self.dynamics_bounds)

    # added
    def get_dynamics_bounds(self):
        return self.dynamics_bounds

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        # added, reset config params
        self.model.body_mass = self.init_params1.copy()
        self.model.geom_friction = self.init_params2.copy()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
