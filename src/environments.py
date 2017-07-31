"""

This module just provides a list of Mujoco environments we support.
It also provides easy indexing into the array of environments, which
is useful for dispatching experiments using environment index numbers.

"""

dynamic_environments = ['InvertedPendulumDynamic-v1',
                        'HalfCheetahDynamic-v1',
                        'HopperDynamic-v1',
                        'Walker2dDynamic-v1']

original_environments = ['InvertedPendulum-v1',
                         'HalfCheetah-v1',
                         'Hopper-v1',
                         'Walker2d-v1']
