import numpy as np
import numba as nb
import scipy as sp

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
]


@nb.experimental.jitclass(spec)
class MetropolisHastings:

    def __init__(self, neu, ned):
        """Metropolis-Hastings random walk.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :return:
        """
        self.neu = neu
        self.ned = ned

    @nb.jit(nopython=True)
    def random_normal_step(self, dX, ne):
        """Random normal distributed step"""
        return np.random.normal(0.0, dX/np.sqrt(3), ne*3).reshape((ne, 3))
11