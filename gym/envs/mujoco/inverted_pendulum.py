import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

        # added
        self.qpos_bounds = [(-1.0, 1.0), (-90.0, 90.0)]

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
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    # added
    def _set_state(self, state):
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:self.model.nq+self.model.nv]
        qpos = self.project_state(qpos, self.qpos_bounds)

        self.set_state(qpos, qvel)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
