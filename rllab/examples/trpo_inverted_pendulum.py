from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from rllab.policies.adversarial_policy import AdversarialPolicy
from rllab.policies.save_policy import saveModel, loadModel

NUM_TRIALS = 5
NUM_ROLLOUTS = 100
NUM_ITERS = 500 # number of training iterations (used to be 250)
TRAIN_ADVERSARIAL = False
ENV_NAME = 'InvertedPendulumDynamic-v1'
ENV_ORG = 'InvertedPendulum-v1'
PROBABILITY = 0.0 # if using Bernoulli adversarial updates, set to probability of adv update
GENERATE_PLOTS = True # if true, save pickle files with the learning curve data
EPS = 0.1
MAX_NORM = True
USE_DYNAMICS = False
RANDOM = False

# get the original state space size first
org_env = normalize(GymEnv(ENV_ORG))
org_env_size = org_env.observation_space.shape[0]
org_env.close()

for trial in range(NUM_TRIALS):
    print('trial {}'.format(trial))
    env = normalize(GymEnv(ENV_NAME))

    policy = AdversarialPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64),
        adaptive_std=False,
        adversarial=TRAIN_ADVERSARIAL,
        eps=EPS,
        probability=PROBABILITY,
        use_dynamics=USE_DYNAMICS,
        random=RANDOM,
        observable_noise=False,
        zero_gradient_cutoff=org_env_size, # zero out gradients except for config params
        use_max_norm=MAX_NORM,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=env.horizon,
        n_itr=NUM_ITERS,
        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
        sampler_args={'n_workers': 2},
        plot_learning_curve=GENERATE_PLOTS,
        trial=trial,
    )
    avg_rewards, std_rewards = algo.train()

    print('trial {}'.format(trial))
    saveModel(algo.policy, 'policy_{}_{}_{}_{}_{}_{}_{}_{}'.format(ENV_NAME, TRAIN_ADVERSARIAL, NUM_ITERS, PROBABILITY, EPS, MAX_NORM, USE_DYNAMICS, trial))

    # save rewards per model over the iterations
    if GENERATE_PLOTS:
        saveModel([range(NUM_ITERS), avg_rewards, std_rewards], 'rewards_{}_{}_{}_{}_{}_{}_{}_{}'.format(ENV_NAME, TRAIN_ADVERSARIAL, NUM_ITERS, PROBABILITY, EPS, MAX_NORM, USE_DYNAMICS, trial))

