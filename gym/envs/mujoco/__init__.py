from gym.envs.mujoco.mujoco_env import MujocoEnv 
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
# added
from gym.envs.mujoco.inverted_pendulum_dynamic import InvertedPendulumDynamicEnv
from gym.envs.mujoco.hopper_dynamic import HopperDynamicEnv
from gym.envs.mujoco.half_cheetah_dynamic import HalfCheetahDynamicEnv
from gym.envs.mujoco.walker2d_dynamic import Walker2dDynamicEnv
