import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.slater import SlaterState_t
from casino.wfn import Gwfn_t, Wfn_t


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
        if self.method == 1 or self.method == 4:
            return self.gibbs_random_step()
        elif self.method == 3:
            return self.simple_random_step()
        return False

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'simple_random_step')
def vmc_simple_random_step(self):
    """Simple random walker with random N-dim gaussian proposal density in
    configuration-by-configuration sampling (CBCS).
    :return: step is accepted
    """

    def impl(self):
        cond = False
        ne = self.wfn.neu + self.wfn.ned
        next_r_e = self.r_e + np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape((ne, 3))
        next_log_value = self.wfn.log_value(next_r_e)[0]
        self.moves += 1
        if 2 * (next_log_value - self.log_value) > np.log(np.random.random()):
            cond, self.r_e, self.log_value = True, next_r_e, next_log_value
            self.accepted += 1
        return cond

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'step_profile')
def vmc_step_profile(self, r_e, e):
    """Position dependent factor on the step size of one electron, what vmc_method 4 adds to EBES.
    The width of a single electron move is set by |grad_i ln psi|, which runs from the nuclear
    charge in a core shell down to order one in the valence, so one global step cannot serve them
    all: at the step that gives the average electron 50%, the valence electrons make up that half
    and a core one is rejected thousands of times in a row. Kato pins |grad ln psi| to Z at the
    nucleus, and between the core and the valence the measured gradient falls as one over the
    distance rather than as the hydrogenic sqrt(Z / r), screened over the Thomas-Fermi length, so
    the branches are Z**2 and (a * Z**(1/3) / r)**2 and meet at r = a / Z**(2/3). The floor is the
    2I of a valence electron, which is also what keeps a bare proton from asking for an infinite
    step. Both constants are fitted to <|grad_i ln psi|**2 | r> measured on neon and argon, see
    examples/step_profile/tabulated.py, and the factor is an estimate of one over that average, so
    step_size carries the units of the kinetic energy sum rule.
    """

    def impl(self, r_e, e):
        gradient = 0.0
        for atom in range(self.wfn.atom_positions.shape[0]):
            charge = self.wfn.atom_charges[atom]
            r = np.sqrt(((r_e[e] - self.wfn.atom_positions[atom]) ** 2).sum())
            screened = 0.815 * charge ** (1 / 3) / r
            gradient = max(gradient, min(charge * charge, screened * screened))
        return 1 / (gradient + 1.577)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'one_electron_step')
def vmc_one_electron_step(self, e):
    """Metropolis step of a single electron, the proposal EBES is built out of.
    Nothing quadratic in the number of electrons is touched: the slater part is a ratio of
    determinants taken against the cached inverse matrices, and the jastrow part is the
    difference of the terms the moved electron takes part in, so the log ratio is accumulated
    rather than recomputed from the whole configuration.
    Under vmc_method 4 the width is that of step_profile at the electron being moved, which makes
    the proposal asymmetric, so the ratio of the densities of the two directions is carried
    alongside the ratio of the wave functions and the returned log ratio is the whole
    Metropolis-Hastings one.
    :param e: electron to move
    :return: step is accepted, ln(psi'**2/psi**2) of the proposal
    """

    def impl(self, e):
        cond = False
        step_size = self.step_size
        if self.method == 4:
            step_size *= self.step_profile(self.r_e, e)
        next_r_e = np.copy(self.r_e)
        next_r_e[e] += np.random.normal(0, np.sqrt(step_size), 3)
        proposal = 0.0
        if self.method == 4:
            next_step_size = self.step_size * self.step_profile(next_r_e, e)
            d2 = ((next_r_e[e] - self.r_e[e]) ** 2).sum()
            proposal = 1.5 * np.log(step_size / next_step_size) + d2 / 2 * (1 / step_size - 1 / next_step_size)
        # backflow spreads a single-electron move over the quasi-particle coordinates of every
        # electron within its cutoff, and a geminal is not a slater determinant, so neither of
        # them leaves one column to update: both recompute the whole configuration instead
        if self.wfn.backflow is None and self.wfn.geminal is None:
            value_log_ratio, _, orbitals, q = self.wfn.value_ratio_1e(self.state, self.n_powers, self.r_e, e, next_r_e[e])
            log_ratio = 2 * value_log_ratio + proposal
            self.moves += 1
            if log_ratio > np.log(np.random.random()):
                cond, self.r_e = True, next_r_e
                self.log_value += value_log_ratio
                self.accepted += 1
                if not self.wfn.accept_1e(self.state, self.n_powers, e, next_r_e[e], orbitals, q):
                    self.state = self.wfn.slater.state(self.wfn._relative_coordinates(next_r_e)[1])
        else:
            next_log_value = self.wfn.log_value(next_r_e)[0]
            log_ratio = 2 * (next_log_value - self.log_value) + proposal
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
        if self.method == 1 or self.method == 4:
            self.state, self.n_powers = self.wfn.caches(self.r_e)
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
    the sum rule, which is exact, from the gaussian shape assumed for it, which is not. Under
    vmc_method 4 the proposal contributes to it as well and what is recorded is the whole
    Metropolis-Hastings ratio, which is the quantity accepted on in any case.
    :param steps: number of steps to walk
    :return: ndarray of log ratios
    """

    def impl(self, steps):
        self.reset()
        log_ratio = np.empty(shape=(steps,))
        ne = self.wfn.neu + self.wfn.ned
        if self.method == 1 or self.method == 4:
            # the electron is taken in turn rather than at random: the sum rule averages over a
            # uniform choice of it, and a sweep covers them uniformly with no extra variance
            for i in range(steps):
                _, log_ratio[i] = self.one_electron_step(i % ne)
        else:
            for i in range(steps):
                next_r_e = self.r_e + np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape((ne, 3))
                next_log_value = self.wfn.log_value(next_r_e)[0]
                log_ratio[i] = 2 * (next_log_value - self.log_value)
                self.moves += 1
                if log_ratio[i] > np.log(np.random.random()):
                    self.r_e, self.log_value = next_r_e, next_log_value
                    self.accepted += 1
        return log_ratio

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(VMC_class_t, 'acceptance_1e')
def vmc_acceptance_1e(self, steps):
    """Acceptance of every electron on its own, which is what vmc_method 4 equalizes and what the
    acceptance of a sweep cannot show: the target is met on average while a core electron sits at
    a fraction of a percent and a valence one close to one, and a coordinate that never moves
    carries no sample whatever the correlation time measured on it says.
    :param steps: number of sweeps to walk
    :return: ndarray of acceptance per electron
    """

    def impl(self, steps):
        self.reset()
        res = np.zeros(shape=(self.wfn.neu + self.wfn.ned,))
        for _ in range(steps):
            for e in range(self.wfn.neu + self.wfn.ned):
                accepted, _ = self.one_electron_step(e)
                res[e] += accepted
        return res / steps

    return impl


def vmc_type(wfn_t):
    """The type of a chain walking a given kind of wave function, which the kind
    of its jastrow makes a type of its own.
    """
    return VMC_class_t(
        [
            ('r_e', nb.float64[:, ::1]),
            ('step_size', nb.float64),
            ('wfn', wfn_t),
            ('method', nb.int64),
            ('log_value', nb.float64),
            ('state', SlaterState_t),
            ('n_powers', nb.float64[:, :, ::1]),
            ('moves', nb.int64),
            ('accepted', nb.int64),
        ]
    )


VMC_t = vmc_type(Wfn_t)
GVMC_t = vmc_type(Gwfn_t)


class VMC(structref.StructRefProxy):
    def __new__(cls, r_e, step_size, wfn, method):
        """Markov chain Monte Carlo.
        :param r_e: initial position
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :param method: vmc method: (1) - EBES, (3) - CBCS, (4) - EBES with the step of step_profile.
        :return:
        """
        vmc_t = GVMC_t if nb.typeof(wfn) == Gwfn_t else VMC_t

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(r_e, step_size, wfn, method):
            self = structref.new(vmc_t)
            self.r_e = r_e
            self.step_size = step_size
            self.wfn = wfn
            self.method = method
            self.log_value = wfn.log_value(r_e)[0]
            self.state, self.n_powers = wfn.caches(r_e)
            self.moves = 0
            self.accepted = 0
            return self

        return init(r_e, step_size, wfn, method)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def r_e(self):
        return self.r_e

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def log_value(self) -> float:
        return self.log_value

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
    def method(self) -> int:
        return self.method

    @method.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def method(self, value):
        self.method = value

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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def acceptance_1e(self, steps):
        return self.acceptance_1e(steps)

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
