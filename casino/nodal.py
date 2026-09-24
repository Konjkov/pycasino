import numpy as np

# configurations of the pilot walk that fixes the epsilon grid, or the single epsilon of an
# optimization; sigma converges far faster than the surface integral it goes on to define
PILOT_STEPS = 10000


def nodal_domain_sums(integrand, epsilon, zeta=0.0, sampled=0.0, jastrow_weight=False, density_weight=False):
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
    :param jastrow_weight: the weight is Φ = J·Π exp(-ζ r_iI), so V_Φ carries the Jastrow's two
        terms beside the two of ζ. The Jastrow cancels out of the reweighting from sampled to zeta,
        being the same factor in both, and only V_Φ knows about it
    :param density_weight: the envelope is the density amplitude Π √n(r_i) rather than the exponent,
        which no ζ can be scanned over: it has the cusp of the nucleus, the decay of the ionization
        energy and the shell structure at once, where one exponent has at most one of the three.
        It replaces the ζ terms of V_Φ rather than joining them, so zeta must be zero with it
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
    if jastrow_weight:
        potential = potential - (integrand[:, 8] - zeta * integrand[:, 9])
    if density_weight:
        potential = potential - integrand[:, 10]
        if jastrow_weight:
            potential = potential - integrand[:, 11]
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


def nodal_domain_gradient_sums(integrand, gradient, epsilon, zeta=0.0, sampled=0.0, log_weight=None):
    """Sums the derivative of E_kin^nda with respect to the parameters is made of, at one tube
    thickness rather than over a grid of them, because an optimization holds epsilon fixed while
    a measurement scans it.

        dF/dp = <K'(σ)·∂σ/∂p> + cov[K(σ), s_p]

    The first term is the node moving under the parameters, the second the measure moving with it,
    s_p being the score ∂ln|Ψ|/∂p of the sampled Φ|Ψ| - Φ carries no parameters and drops out.
    What is summed is K(σ) = 3(ε - σ)/ε³ inside the tube, so K' is the constant -3/ε³ there, and
    with ∂σ/∂p = -σ³/2 · ∂|∇lnΨ|²/∂p the first term collects 3σ³/2ε³ against that derivative.

    :param integrand: as nodal_domain_sums takes it - array(nconfig, 8)
    :param gradient: the score and the derivative of |∇lnΨ|² of the configurations inside the tube,
        in their order - array(inside, 2, parameters). Outside it the kernel and its derivative
        both vanish, so the costly derivative is never needed there
    :param epsilon: half-thickness of the tube, in bohr
    :param zeta: exponent of the one-particle weight the average is taken with
    :param sampled: exponent the chain walked
    :param log_weight: log|Ψ_p| - log|Ψ_p0| of every configuration, when the sample was drawn at
        one set of parameters and the average is being taken at another. The expression above is
        unchanged by it - the sampled measure cancels out of every ratio and what is left is the
        score of Ψ_p, which is what gradient carries - so an optimization can walk once and then
        move over the fixed sample with an exact derivative rather than an approximate one
    :return: array(2) of the surface sum and the overlap sum, and array(2, parameters) of the node
        term and the first half of the covariance. The second half, the weighted sum of the score,
        runs over every configuration and is left to the caller, so that nothing of that size is
        ever held
    """
    sigma = 1 / np.sqrt(integrand[:, 1])
    if zeta != sampled:
        weight = np.exp(-(zeta - sampled) * integrand[:, 3])
    else:
        weight = np.ones(shape=sigma.shape)
    if log_weight is not None:
        weight = weight * np.exp(log_weight)
    inside = tube(integrand, epsilon)
    sigma, inner = sigma[inside], weight[inside]
    value = 3 * (epsilon - sigma) / epsilon**3
    node = 3 * sigma**3 / (2 * epsilon**3)
    scalars = np.array([inner @ value, weight.sum()])
    vectors = np.stack(((inner * node) @ gradient[:, 1], (inner * value) @ gradient[:, 0]))
    return scalars, vectors


def tube(integrand, epsilon):
    return 1 / np.sqrt(integrand[:, 1]) < epsilon
