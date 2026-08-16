import numpy as np

"""
https://stats.stackexchange.com/questions/459130/determining-standard-error-of-the-mean-from-a-correlated-stationary-time-series
https://ljmartin.github.io/technical-notes/stats/estimators-autocorrelated/

There are many different solutions to calculating the SEM for autocorrelated data. There's no "true"
solution, meaning each one was developed in a different field for a different purpose. They all have
a bit of overlap, but differ in their ease of use. Ideally we want something that is robust to as many
different situations as possible, giving SEM's that include the true mean value at the right rate.

1. Block averaging
2. Estimating neff from the autocorrelation function
3. Autoregressive processes AR(1) estimation
4. AR(1) Bayesian estimation
"""


def correlation_time(x, c=3):
    """Integrated autocorrelation time, i.e. the factor by which the variance of the mean exceeds
    var(x) / n. The window is the first w with w >= c * tau, which keeps the bias of truncating
    below the noise of the tail whatever the shape of the decay, so unlike an autoregressive fit
    it assumes neither a single exponential nor any other model of the series.
    :param x: time series
    :param c: window constant, 3 as in CASINO's correlation_time
    :return: tau, error in tau, both negative when the series is too short to say anything
    """
    n = x.size
    if n < 10:
        return -1.0, -1.0
    d = x - x.mean()
    variance = d @ d / n
    if variance <= 0:
        return -1.0, -1.0
    tau, w = 1.0, 1
    for w in range(1, n):
        tau += 2 * (d[: n - w] @ d[w:]) / (n - w) / variance
        if w >= round(c * tau) or w == n - 1:
            break
    return tau, tau * np.sqrt((4 * w + 2) / n)


class Reblock:
    """On-the-fly reblocking. Every block length 2**k keeps the number, the sum and the sum of
    squares of the block means closed so far, plus the single mean waiting for its partner, so
    what is stored grows with the logarithm of the number of steps rather than with the number
    of steps itself and the series is never held.
    """

    def __init__(self):
        self.nstep = 0
        self.count = []
        self.total = []
        self.total2 = []
        self.carry = []

    def add(self, data):
        """Accumulate a chunk of the series. A block mean of one level is a sample of the next,
        so the chunk halves as it goes up and the whole call costs twice its length.
        :param data: consecutive values of the series
        """
        x = np.asarray(data, dtype=float)
        self.nstep += x.size
        k = 0
        while x.size:
            if k == len(self.count):
                self.count.append(0)
                self.total.append(0.0)
                self.total2.append(0.0)
                self.carry.append(None)
            self.count[k] += x.size
            self.total[k] += x.sum()
            self.total2[k] += x @ x
            if self.carry[k] is not None:
                x = np.concatenate((np.array([self.carry[k]]), x))
            if x.size % 2:
                self.carry[k] = x[-1]
                x = x[:-1]
            else:
                self.carry[k] = None
            x = (x[0::2] + x[1::2]) / 2
            k += 1

    def stderr(self):
        """Standard error of the mean at the block length picked by the criterion of
        U. Wolff, Comput. Phys. Commun. 156, 143 (2004), Eq. (47): the smallest block length with
        B**3 >= 2 * nstep * ncorr(B)**2, at which the systematic error of the error bar is half
        of its statistical error. Falls back to the unreblocked value when no block length
        qualifies, i.e. when the run is too short for the plateau to have set in.
        """
        err = np.zeros(shape=len(self.count))
        for k in range(len(self.count)):
            n = self.count[k]
            if 2 ** (k + 1) > self.nstep or n < 2:
                break
            variance = (self.total2[k] - self.total[k] ** 2 / n) / (n - 1)
            err[k] = np.sqrt(max(variance, 0) / n)
        if err.size == 0 or err[0] == 0:
            return 0.0
        for k in range(err.size):
            if err[k] == 0:
                break
            ncorr = (err[k] / err[0]) ** 2
            if 8.0**k >= 2 * self.nstep * ncorr**2:
                return err[k]
        return err[0]


def correlated_sem(energy):
    """Block averaging standard error of the mean (SEM)
    :param energy: one series, or one series per row
    :return: SEM of the series, or an array of the SEM of every row
    """
    energy = np.asarray(energy)
    if energy.ndim == 1:
        reblock = Reblock()
        reblock.add(energy)
        return reblock.stderr()
    if energy.shape[0] == 1:
        return correlated_sem(energy[0])
    return np.array([correlated_sem(row) for row in energy])
