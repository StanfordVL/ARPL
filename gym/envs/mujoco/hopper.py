import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        # added
        self.qpos_bounds = [(None, None), (None, None), (None, None), 
                            (-150.0, 0.0), (-150.0, 0.0), (-45.0, 45.0)]

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

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10)
        ])

    # added
    def _set_state(self, state):
        # first component of qpos is fixed / unobservable
        # print('qpos')
        # print(self.model.data.qpos)
        # print('qvel')
        # print(self.model.data.qvel)
        qpos = np.concatenate([[self.model.data.qpos.flat[0]], state[:self.model.nq-1]])
        qvel = np.clip(state[self.model.nq-1:],-10,10)
        qpos = self.project_state(qpos, self.qpos_bounds)
        self.set_state(qpos, qvel)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
