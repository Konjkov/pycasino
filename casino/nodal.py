import numpy as np


def nodal_domain_sums(integrand, epsilon, zeta=0.0, sampled=0.0):
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

    The kernel is δ_ε(σ) = 3|σ|(ε - |σ|)/ε³ for |σ| < ε, whose factor |σ| cancels the |∇Ψ|/|Ψ| the
    change of variable brings, leaving 3(ε - |σ|)/ε³ to be summed: bounded by 3/ε², and carrying no
    |∇lnΨ|, which is what keeps the estimator from inheriting that quantity's variance near the
    node. |∇σ| = 1 on the node itself and is not evaluated off it, which leaves an O(ε) bias —
    visible in the scan over ε as the departure from the plateau.

    The density of σ under Φ|Ψ| vanishes linearly at the node, so what a kernel has to satisfy is
    ∫δ_ε = 1 and nothing more; every normalized kernel returns the same leading term. The plain
    |σ|/ε², which reduces the sum to a count of the configurations in the tube, satisfies it too
    and was what this used. It is dropped because a count is discontinuous in the parameters of Ψ
    and so has no gradient, while this kernel goes to zero at the tube edge and is differentiable
    there. What it costs is 3/2 in variance, 1.22 in the error bar; what it gains beside the
    gradient is a quarter off the O(ε) bias, whose coefficient goes 2/3 → 1/2.

    The weight carries the configurations from the measure the chain sampled, Φ_sampled·|Ψ|, to
    the one the estimator is written on, Φ_ζ·|Ψ|, so it is exp(-(ζ - sampled)·Σ r_iI) and is one
    when the chain already walked the measure being estimated. ζ = 0 is the constant weight, for
    which Φ = 1 and V_Φ = 0.

    Reweighting from below, sampled < ζ, is safe in that the weights are bounded above and no
    configuration can dominate the sums, and what it costs is the effective sample size (Σw)²/Σw²,
    returned along with them. What it cannot do is invent a sample: if Φ_ζ|Ψ| is large where the
    chain never went, no weight recovers it, the ESS stays high because the weights are uniformly
    tiny, and the answer comes out of the wrong region entirely. That is what a walk carrying its
    own ζ is for.

    :param integrand: log|Ψ|, |∇Ψ/Ψ|², V, Σ r_iI, Σ 1/r_iI, Σ_i |Σ_I r̂_iI|² and the e-e and e-n
        parts of V, of every configuration - array(nconfig, 8)
    :param epsilon: tube half-thicknesses, in bohr
    :param zeta: exponent of the one-particle weight the average is taken with, in inverse bohr
    :param sampled: exponent the chain itself walked, equal to zeta for a walk of Φ|Ψ| and zero
        for a walk of |Ψ|
    :return: array(epsilon.size, 3) of the surface sum, its sum of squares and the number of
        configurations inside the tube, and array(8) of the number of configurations, the overlap
        sum, the sums of V - V_Φ and (V - V_Φ)² over it, the sum of the squared weights, the sums
        of the e-e and e-n parts and the sum of Σ r_iI, which is the scale ζ is read against
    """
    sigma = 1 / np.sqrt(integrand[:, 1])
    if zeta != sampled:
        # up to the normalization of Φ, which every ratio taken here divides out
        weight = np.exp(-(zeta - sampled) * integrand[:, 3])
    else:
        weight = np.ones(shape=sigma.shape)
    # V_Φ belongs to the average being taken and not to the measure it was reached from, so it
    # goes by zeta alone, whether the chain walked Φ|Ψ| or the weight brought it there
    if zeta:
        potential = integrand[:, 2] - (zeta**2 * integrand[:, 5] / 2 - zeta * integrand[:, 4])
    else:
        potential = integrand[:, 2]
    surface = np.empty(shape=(epsilon.size, 3))
    for i, eps in enumerate(epsilon):
        inside = sigma < eps
        value = np.where(inside, 3 * weight * (eps - sigma) / eps**3, 0.0)
        surface[i] = value.sum(), value @ value, inside.sum()
    overlap = np.array(
        [
            sigma.size,
            weight.sum(),
            weight @ potential,
            weight @ (potential * potential),
            weight @ weight,
            weight @ integrand[:, 6],
            weight @ integrand[:, 7],
            weight @ integrand[:, 3],
        ]
    )
    return surface, overlap
