import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.slater import SlaterState_t
from casino.wfn import Wfn_t


@structref.register
class VMC_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'random_step')
def vmc_random_step(self):
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
@overload_method(VMC_class_t, 'simple_random_step')
def vmc_simple_random_step(self):
    """Simple random walker with random N-dim square proposal density in
    configuration-by-configuration sampling (CBCS).
    :return: step is accepted
    """

    def impl(self):
        cond = False
        ne = self.wfn.neu + self.wfn.ned
        next_r_e = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        next_log_value = self.wfn.log_value(next_r_e)[0]
        self.moves += 1
        if 2 * (next_log_value - self.log_value) > np.log(np.random.random()):
            cond, self.r_e, self.log_value = True, next_r_e, next_log_value
            self.accepted += 1
        return cond

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'one_electron_step')
def vmc_one_electron_step(self, e):
    """Metropolis step of a single electron, the proposal EBES is built out of.
    Nothing quadratic in the number of electrons is touched: the slater part is a ratio of
    determinants taken against the cached inverse matrices, and the jastrow part is the
    difference of the terms the moved electron takes part in, so the log ratio is accumulated
    rather than recomputed from the whole configuration.
    :param e: electron to move
    :return: step is accepted, ln(psi'**2/psi**2) of the proposal
    """

    def impl(self, e):
        cond = False
        next_r_e = np.copy(self.r_e)
        next_r_e[e] += self.step_size * np.random.uniform(-1, 1, 3)
        # backflow spreads a single-electron move over the quasi-particle coordinates of every
        # electron within its cutoff, and a geminal is not a slater determinant, so neither of
        # them leaves one column to update: both recompute the whole configuration instead
        if self.wfn.backflow is None and self.wfn.geminal is None:
            # the determinant needs the nuclear distances of the moved electron and of nobody
            # else, and that column is cheaper to build than to look up in an array of all of them
            next_log_value, _, orbitals, q = self.state.ratio_1e(next_r_e[e] - self.wfn.atom_positions, e)
            log_ratio = 2 * (next_log_value - self.state.log_value)
            if self.wfn.jastrow is not None:
                # the nuclear distances of the electrons that stayed put are common to both ends
                # of the proposal, so their powers are built once and only the row of the moved
                # electron is replaced between the two evaluations
                e_vectors_1e, n_vectors = self.wfn._relative_coordinates_1e(self.r_e, e)
                n_powers = self.wfn.jastrow.en_powers(n_vectors)
                jastrow_value = self.wfn.jastrow.value_1e(e_vectors_1e, n_powers, e)
                self.wfn.jastrow.update_en_powers_1e(n_powers, next_r_e[e] - self.wfn.atom_positions, e)
                log_ratio += 2 * (self.wfn.jastrow.value_1e(next_r_e[e] - next_r_e, n_powers, e) - jastrow_value)
            self.moves += 1
            if log_ratio > np.log(np.random.random()):
                cond, self.r_e = True, next_r_e
                self.log_value += log_ratio / 2
                self.accepted += 1
                if not self.state.accept_1e(e, orbitals, q):
                    self.state = self.wfn.slater.state(self.wfn._relative_coordinates(next_r_e)[1])
        else:
            next_log_value = self.wfn.log_value(next_r_e)[0]
            log_ratio = 2 * (next_log_value - self.log_value)
            self.moves += 1
            if log_ratio > np.log(np.random.random()):
                cond, self.r_e, self.log_value = True, next_r_e, next_log_value
                self.accepted += 1
        return cond, log_ratio

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'gibbs_random_step')
def vmc_gibbs_random_step(self):
    """Simple random walker with electron-by-electron sampling (EBES)
    :return: step is accepted
    """

    def impl(self):
        cond = False
        for i in range(self.wfn.neu + self.wfn.ned):
            accepted, _ = self.one_electron_step(i)
            cond |= accepted
        return cond

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'reset')
def vmc_reset(self):
    """Bring the cached wave function up to date with the configuration and start counting
    moves afresh. The walker outlives both the configuration and the parameters it was built
    with, the latter changing under it on every optimization cycle.
    """

    def impl(self):
        self.log_value = self.wfn.log_value(self.r_e)[0]
        if self.method == 1:
            self.state = self.wfn.slater.state(self.wfn._relative_coordinates(self.r_e)[1])
        self.moves = 0
        self.accepted = 0

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'random_walk')
def vmc_random_walk(self, steps, decorr_period):
    """Metropolis-Hastings random walk.
    :param steps: number of steps to walk
    :param decorr_period: decorrelation period
    :return: ndarray of electron positions
    """

    def impl(self, steps, decorr_period):
        self.reset()
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


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'log_ratio_walk')
def vmc_log_ratio_walk(self, steps):
    """Metropolis-Hastings random walk recording ln(psi'**2/psi**2) for every proposed move,
    one whole configuration at a time in CBCS and one electron at a time in EBES. That is the
    quantity the VMC step size sum rule is a statement about, and measuring it directly separates
    the sum rule, which is exact, from the gaussian shape assumed for it, which is not.
    :param steps: number of steps to walk
    :return: ndarray of log ratios
    """

    def impl(self, steps):
        self.reset()
        log_ratio = np.empty(shape=(steps,))
        ne = self.wfn.neu + self.wfn.ned
        if self.method == 1:
            # the electron is taken in turn rather than at random: the sum rule averages over a
            # uniform choice of it, and a sweep covers them uniformly with no extra variance
            for i in range(steps):
                _, log_ratio[i] = self.one_electron_step(i % ne)
        else:
            for i in range(steps):
                next_r_e = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
                next_log_value = self.wfn.log_value(next_r_e)[0]
                log_ratio[i] = 2 * (next_log_value - self.log_value)
                self.moves += 1
                if log_ratio[i] > np.log(np.random.random()):
                    self.r_e, self.log_value = next_r_e, next_log_value
                    self.accepted += 1
        return log_ratio

    return impl


VMC_t = VMC_class_t(
    [
        ('r_e', nb.float64[:, ::1]),
        ('step_size', nb.float64),
        ('wfn', Wfn_t),
        ('method', nb.int64),
        ('log_value', nb.float64),
        ('state', SlaterState_t),
        ('moves', nb.int64),
        ('accepted', nb.int64),
    ]
)


class VMC(structref.StructRefProxy):
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
            self = structref.new(VMC_t)
            self.r_e = r_e
            self.step_size = step_size
            self.wfn = wfn
            self.method = method
            self.log_value = wfn.log_value(r_e)[0]
            self.state = wfn.slater.state(wfn._relative_coordinates(r_e)[1])
            self.moves = 0
            self.accepted = 0
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

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def acceptance(self) -> float:
        """Fraction of the proposals of the last walk that were accepted. One proposal is one
        electron in EBES and the whole configuration in CBCS, so this is what the 50% rule and
        the step size sum rule are both stated in terms of.
        """
        return self.accepted / self.moves

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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def log_ratio_walk(self, steps):
        return self.log_ratio_walk(steps)

    @staticmethod
    def observable(observable, position):
        """VMC observable.
        :param observable: observable function
        :param position: random walk positions
        :return: observable values
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


structref.define_boxing(VMC_class_t, VMC)
