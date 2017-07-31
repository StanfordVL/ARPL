import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        # added
        self.qpos_bounds = [(None, None), (None, None), (None, None), 
                            (-0.52, 1.05), (-0.785, 0.785), (-0.4, 0.785),
                            (-1.0, 0.7), (-1.2, 0.87), (-0.5, 0.5)]

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

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    # added
    def _set_state(self, state):
        # first component of qpos is fixed / unobservable
        # print('qpos')
        # print(self.model.data.qpos)
        # print('qvel')
        # print(self.model.data.qvel)
        qpos = np.concatenate([[self.model.data.qpos.flat[0]], state[:self.model.nq-1]])
        qvel = state[self.model.nq-1:]
        qpos = self.project_state(qpos, self.qpos_bounds)
        self.set_state(qpos, qvel)
        #print('internal state')
        #print(self._get_obs())

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
