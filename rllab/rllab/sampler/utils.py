import numpy as np
from rllab.misc import tensor_utils
import time

from rllab.policies.adversarial_policy import AdversarialPolicy
from rllab.policies.curriculum_policy import CurriculumPolicy
from rllab.policies.curriculum_policy import CurriculumPolicy
from rllab.policies.model_free_adversarial_policy import ModelFreeAdversarialPolicy
from rllab.policies.deterministic_mlp_curriculum_policy import DeterministicMLPCurriculumPolicy

# note that curriculum is passed in due to scoping issues in accessing the agent's curriculum
def rollout(env, agent, max_path_length=np.inf, curriculum=None, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # We only support adversarial policies ;)
    assert(isinstance(agent, AdversarialPolicy) or isinstance(agent, CurriculumPolicy) or isinstance(agent, DeterministicMLPCurriculumPolicy))
    agent.reset()

    # If doing curriculum learning, draw a configuration at random and set it
    # for the remainder of the episode.
    if isinstance(agent, CurriculumPolicy) or isinstance(agent, DeterministicMLPCurriculumPolicy):
        assert(curriculum is not None)
        agent.sample_from_curriculum(curriculum)

    # keep track of control effort
    action_norms_squared = []

    # record qpos and qvel if necessary
    qpos_traj = None
    qvel_traj = None

    # Draw initial state from the environment.
    o = env.reset()
    # Set initial dynamics, if necessary.
    if agent.set_dynamics is not None:
        o = agent.set_adversarial_dynamics(env=env, full_state=o, dynamics=agent.set_dynamics)

    # Apply adversarial update, if necessary.
    o = agent.do_perturbation(env=env, state=o, is_start=True, scale=None)
    #o = agent.do_perturbation(env=env, state=o, is_start=True, scale=1e-3)
    # Record the trajectory, if necessary.
    if agent.record_traj:
        qpos_traj = []
        qvel_traj = []
        (nq, nv) = env.qpos_dim()
        st_vec = env.state_vector()
        qpos_traj.append(st_vec[:nq])
        qvel_traj.append(st_vec[nq:])
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a_flat = env.action_space.flatten(a)
        action_norms_squared.append(np.sum(a_flat * a_flat))
        next_o, r, d, env_info = env.step(a)

        # Apply adversarial update, if necessary.
        next_o = agent.do_perturbation(env=env, state=next_o, is_start=False, scale=None)
        #next_o = agent.do_perturbation(env=env, state=next_o, is_start=False, scale=1e-3)

        # Record the trajectory, if necessary
        if agent.record_traj:
            st_vec = env.state_vector()
            qpos_traj.append(st_vec[:nq])
            qvel_traj.append(st_vec[nq:])

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    # mask unneeded observations if necessary
    if agent.mask_augmentation:
        observations = [x[:agent.zero_gradient_cutoff] for x in observations]

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        qpos_traj=qpos_traj,
        qvel_traj=qvel_traj,
        control_norms=action_norms_squared,
    )
