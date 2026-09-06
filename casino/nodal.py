import numpy as np


def nodal_domain_sums(integrand, epsilon, zeta=0.0):
    """Sums that the weighted nodal domain averages of Mitas & Annaberdiyev (arXiv:2109.01734)
    are made of, for every tube half-thickness at once.

    E_nda(Φ) = ∫_∂Ω Φ|∇Ψ|dS / ∫ Φ|Ψ|dR + ∫ VΦ|Ψ|dR / ∫ Φ|Ψ|dR + ∫ |Ψ|T_kin Φ dR / ∫ Φ|Ψ|dR

    the last term vanishing for a constant weight Φ. The nodal hypersurface integral is taken by
    the co-area formula, with the delta smeared in σ = Ψ/|∇Ψ| — the first order distance to the
    node, a length. Nothing of the geometry of the node is needed then, no point is projected onto
    it, and there are no focal points to condition.

    Smearing in the value of Ψ instead, which is the same identity written with δ(Ψ)|∇Ψ|², is
    what the estimator must NOT do for an atom: |Ψ| is small in the whole exponential tail of
    configuration space and not only at the node, so the tube fills up with configurations that
    are merely far away and the limit is approached too slowly to be of use. σ stays of order one
    out there, |Ψ| and |∇Ψ| vanishing together, and the tube holds the node alone.

    The kernel is δ_ε(σ) = |σ|/ε² for |σ| < ε, whose factor |σ| cancels the |∇Ψ|/|Ψ| the change of
    variable brings, so that with configurations distributed as Φ|Ψ| the surface integral is the
    occupancy of the tube over ε²: bounded, and Poisson in the number of configurations it holds.
    |∇σ| = 1 on the node itself and is not evaluated off it, which leaves an O(ε) bias — visible
    in the scan over ε as the departure from the plateau, along with the O(ε²) of the kernel.

    The configurations are the walk's own, distributed as |Ψ|, and the weight Φ = exp(-ζ Σ r_iI)
    carries them to the measure Φ|Ψ| the estimator is written on. Reweighting is safe in this
    direction and only in this one: Φ ≤ 1, so the weights are bounded above and no configuration
    can dominate the sums. What it can cost is the effective sample size (Σw)²/Σw², which is
    returned along with them. ζ = 0 is the constant weight, for which Φ = 1 and V_Φ = 0.

    :param integrand: log|Ψ|, |∇Ψ/Ψ|², V, Σ r_iI, Σ 1/r_iI and Σ_i |Σ_I r̂_iI|² of every
        configuration - array(nconfig, 6), of which only the first three are read at ζ = 0
    :param epsilon: tube half-thicknesses, in bohr
    :param zeta: exponent of the one-particle weight, in inverse bohr
    :return: array(epsilon.size, 3) of the surface sum, its sum of squares and the number of
        configurations inside the tube, and array(5) of the number of configurations, the overlap
        sum, the sums of V - V_Φ and (V - V_Φ)² over it and the sum of the squared weights
    """
    sigma = 1 / np.sqrt(integrand[:, 1])
    if zeta:
        # exp(-ζ Σ r_iI) against the walk's own |Ψ|, up to the normalization the ratios divide out
        weight = np.exp(-zeta * integrand[:, 3])
        potential = integrand[:, 2] - (zeta**2 * integrand[:, 5] / 2 - zeta * integrand[:, 4])
    else:
        weight = np.ones(shape=sigma.shape)
        potential = integrand[:, 2]
    surface = np.empty(shape=(epsilon.size, 3))
    for i, eps in enumerate(epsilon):
        inside = sigma < eps
        value = np.where(inside, weight / eps**2, 0.0)
        surface[i] = value.sum(), value @ value, inside.sum()
    overlap = np.array([sigma.size, weight.sum(), weight @ potential, weight @ (potential * potential), weight @ weight])
    return surface, overlap
