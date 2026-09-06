import numpy as np


def nodal_domain_sums(integrand, epsilon, log_weight=None):
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

    :param integrand: log|Ψ|, |∇Ψ/Ψ|² and V of every configuration - array(nconfig, 3)
    :param epsilon: tube half-thicknesses, in bohr
    :param log_weight: log of Φ/p up to a constant, p being the density the configurations are
        sampled from. Defaults to configurations distributed as Φ|Ψ| itself.
    :return: array(epsilon.size, 3) of the surface sum, its sum of squares and the number of
        configurations inside the tube, and array(4) of the number of configurations, the overlap
        sum and the sums of V and V² over it
    """
    sigma = 1 / np.sqrt(integrand[:, 1])
    if log_weight is None:
        weight = np.ones(shape=sigma.shape)
    else:
        weight = np.exp(log_weight + integrand[:, 0])
    surface = np.empty(shape=(epsilon.size, 3))
    for i, eps in enumerate(epsilon):
        inside = sigma < eps
        value = np.where(inside, weight / eps**2, 0.0)
        surface[i] = value.sum(), value @ value, inside.sum()
    potential = integrand[:, 2]
    overlap = np.array([sigma.size, weight.sum(), weight @ potential, weight @ (potential * potential)])
    return surface, overlap
