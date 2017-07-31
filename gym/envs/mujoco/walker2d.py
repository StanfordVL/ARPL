import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

        # added
        self.qpos_bounds = [(None, None), (None, None), (None, None), 
                            (-150.0, 0.0), (-150.0, 0.0), (-45.0, 45.0),
                            (-150.0, 0.0), (-150.0, 0.0), (-45.0, 45.0)]

    def _step(self, a):
        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt )
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0
                    and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel,-10,10)]).ravel()

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
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
