from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class LinearFeatureQ(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.n_actions = env_spec.action_space.shape[0]

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        a = path["actions"]
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        # return np.concatenate([o, o ** 2, a, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)
        return np.concatenate([a, o], axis=1)

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    @overrides
    def predict(self, path):
        # print('path for baseline: ', path)
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        # print('features shape', self._features(path).shape)
        # print('features for baseline', self._features(path))
        return self._features(path).dot(self._coeffs)

    def gradient(self, o, a):
        if self._coeffs is None:
            return np.zeros(self.n_actions)
        return self._coeffs[0:self.n_actions]
