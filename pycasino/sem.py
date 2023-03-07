import pyblock

"""
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


def correlated_sem(energy):
    """Block averaging standard error of the mean (SEM)"""
    if len(energy.shape) == 1:
        steps = energy.shape[0]
    elif energy.shape[0] == 1:
        energy = energy[0]
        steps = energy.shape[0]
    else:
        steps = energy.shape[1]
    reblock_data = pyblock.blocking.reblock(energy)
    opt = pyblock.blocking.find_optimal_block(steps, reblock_data)
    if opt[0]:
        try:
            return reblock_data[opt[0]].std_err
        except TypeError:
            print(opt, energy.mean(), energy.var())
            try:
                return reblock_data[int(opt[0])].std_err
            except ValueError:
                return 0
