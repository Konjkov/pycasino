import numpy as np
import numba as nb
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method


@structref.register
class VMCMarkovChain_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


VMCMarkovChain_instance_t = VMCMarkovChain_class_t([
    ('r_e', nb.float64[:, :]),
    ('cond', nb.int64),
    ('step_size', nb.float64),
    ('wfn', Wfn.class_type.instance_type),
    ('method', nb.int64),
    ('probability_density', nb.float64),
])


class VMCMarkovChain(structref.StructRefProxy):

    def __new__(cls, r_e, step_size, wfn, method):
        """Markov chain Monte Carlo.
        :param r_e: initial position
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :param method: vmc method: (1) - EBES (work in progress), (3) - CBCS.
        :return:
        """
        cond = 0
        probability_density = wfn.value(r_e) ** 2
        return structref.StructRefProxy.__new__(cls, r_e, cond, step_size, wfn, method, probability_density)

    def random_walk(self, steps, decorr_period):
        return random_walk(self, steps, decorr_period)


@nb.njit(nogil=True, parallel=False, cache=True)
def random_walk(self, steps, decorr_period):
    """Metropolis-Hastings random walk.
    :param steps: number of steps to walk
    :param decorr_period: decorrelation period
    :return:
    """
    self.probability_density = self.wfn.value(self.r_e) ** 2
    condition = np.empty(shape=(steps,), dtype=np.int_)
    position = np.empty(shape=(steps,) + self.r_e.shape)

    for i in range(steps):
        cond = 0
        for _ in range(decorr_period):
            self.random_step()
            cond += self.cond
        condition[i], position[i] = cond, self.r_e

    return condition, position


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'random_step')
def vmcmarkovchain_random_step(self):
    """Wrapper"""
    def impl(self):
        if self.method == 1:
            self.gibbs_random_step()
        elif self.method == 3:
            self.simple_random_step()
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'simple_random_step')
def vmcmarkovchain_simple_random_step(self):
    """Simple random walker with random N-dim square proposal density in
    configuration-by-configuration sampling (CBCS).
    """
    def impl(self):
        ne = self.wfn.neu + self.wfn.ned
        next_r_e = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        next_probability_density = self.wfn.value(next_r_e) ** 2
        self.cond = next_probability_density / self.probability_density > np.random.random()
        if self.cond:
            self.r_e, self.probability_density = next_r_e, next_probability_density
    return impl

@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'gibbs_random_step')
def vmcmarkovchain_gibbs_random_step(self):
    """Simple random walker with electron-by-electron sampling (EBES)
    """
    def impl(self):
        ne = self.wfn.neu + self.wfn.ned
        self.cond = 0
        for i in range(ne):
            next_r_e = np.copy(self.r_e)
            next_r_e[i] += self.step_size * np.random.uniform(-1, 1, 3)
            next_probability_density = self.wfn.value(next_r_e) ** 2
            cond = next_probability_density / self.probability_density > np.random.random()
            self.cond += cond
            if cond:
                self.r_e, self.probability_density = next_r_e, next_probability_density
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMCMarkovChain_class_t, 'vmc_energy')
def vmcmarkovchain_vmc_energy(self, condition, position):
    """VMC energy.
    :param condition: accept/reject conditions
    :param position: random walk positions
    :return:
    """
    def impl(self, condition, position):
        # FIXME: very slow
        first_res = self.wfn.energy(position[0])
        res = np.empty(shape=condition.shape + np.shape(first_res))
        res[0] = first_res

        for i in range(1, condition.shape[0]):
            if condition[i]:
                res[i] = self.wfn.energy(position[i])
            else:
                res[i] = res[i-1]
        return res
    return impl


# This associates the proxy with MyStruct_t for the given set of fields.
# Notice how we are not constraining the type of each field.
# Field types remain generic.
structref.define_proxy(VMCMarkovChain, VMCMarkovChain_class_t, ['r_e', 'cond', 'step_size', 'wfn', 'method', 'probability_density'])
