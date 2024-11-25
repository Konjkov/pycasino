import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.wfn import Wfn_t


@structref.register
class VMCMarkovChain_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'random_step')
def vmcmarkovchain_random_step(self):
    """VMC random step wrapper.
    :return: step is accepted
    """

    def impl(self):
        if self.method == 1:
            return self.gibbs_random_step()
        elif self.method == 3:
            return self.simple_random_step()
        return False

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'simple_random_step')
def vmcmarkovchain_simple_random_step(self):
    """Simple random walker with random N-dim square proposal density in
    configuration-by-configuration sampling (CBCS).
    :return: step is accepted
    """

    def impl(self):
        cond = False
        ne = self.wfn.neu + self.wfn.ned
        next_r_e = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        next_probability_density = self.wfn.value(next_r_e) ** 2
        if next_probability_density / self.probability_density > np.random.random():
            cond, self.r_e, self.probability_density = True, next_r_e, next_probability_density
        return cond

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'gibbs_random_step')
def vmcmarkovchain_gibbs_random_step(self):
    """Simple random walker with electron-by-electron sampling (EBES)
    :return: step is accepted
    """

    def impl(self):
        cond = False
        ne = self.wfn.neu + self.wfn.ned
        for i in range(ne):
            next_r_e = np.copy(self.r_e)
            next_r_e[i] += self.step_size * np.random.uniform(-1, 1, 3)
            next_probability_density = self.wfn.value(next_r_e) ** 2
            if next_probability_density / self.probability_density > np.random.random():
                cond, self.r_e, self.probability_density = True, next_r_e, next_probability_density
        return cond

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'random_walk')
def vmcmarkovchain_random_walk(self, steps, decorr_period):
    """Metropolis-Hastings random walk.
    :param steps: number of steps to walk
    :param decorr_period: decorrelation period
    :return:
    """

    def impl(self, steps, decorr_period):
        self.probability_density = self.wfn.value(self.r_e) ** 2
        position = np.full(shape=(steps,) + self.r_e.shape, fill_value=np.nan)
        # the following value will be rewritten as the first step is taken
        position[0] = self.r_e

        for i in range(steps):
            cond = False
            for _ in range(decorr_period):
                cond |= self.random_step()
            if cond:
                position[i] = self.r_e

        return position

    return impl


VMCMarkovChain_t = VMCMarkovChain_class_t(
    [
        ('r_e', nb.float64[:, ::1]),
        ('step_size', nb.float64),
        ('wfn', Wfn_t),
        ('method', nb.int64),
        ('probability_density', nb.float64),
    ]
)


class VMCMarkovChain(structref.StructRefProxy):
    def __new__(cls, *args, **kwargs):
        """Markov chain Monte Carlo.
        :param r_e: initial position
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :param method: vmc method: (1) - EBES (work in progress), (3) - CBCS.
        :return:
        """

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(r_e, step_size, wfn, method):
            self = structref.new(VMCMarkovChain_t)
            self.r_e = r_e
            self.step_size = step_size
            self.wfn = wfn
            self.method = method
            self.probability_density = wfn.value(r_e) ** 2
            return self

        return init(*args, **kwargs)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def step_size(self) -> float:
        return self.step_size

    @step_size.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def step_size(self, value):
        self.step_size = value

    def bbk_random_step(self):
        """Brünger–Brooks–Karplus (13 B. Brünger, C. L. Brooks, and M. Karplus, Chem. Phys. Lett. 105, 495 1984)."""
        raise NotImplementedError

    def force_interpolation_random_step(self):
        """M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids Oxford University Press, Oxford, 1989 and references in Sec. 9.3."""
        raise NotImplementedError

    def splitting_random_step(self):
        """J. A. Izaguirre, D. P. Catarello, J. M. Wozniak, and R. D. Skeel, J. Chem. Phys. 114, 2090 2001."""
        raise NotImplementedError

    def ricci_ciccotti_random_step(self):
        """A. Ricci and G. Ciccotti, Mol. Phys. 101, 1927 2003."""
        raise NotImplementedError

    @nb.njit(nogil=True, parallel=False, cache=True)
    def random_walk(self, steps, decorr_period):
        return self.random_walk(steps, decorr_period)


structref.define_boxing(VMCMarkovChain_class_t, VMCMarkovChain)


def vmc_observable(position, observable):
    """VMC observable.
    :param position: random walk positions
    :param observable: observable function
    :return:
    """
    res_0 = observable(position[0])
    res = np.empty(shape=position.shape[:1] + np.shape(res_0))
    res[0] = res_0

    for i in range(1, position.shape[0]):
        if np.isnan(position[i, 0, 0]):
            res[i] = res[i - 1]
        else:
            res[i] = observable(position[i])
    return res
