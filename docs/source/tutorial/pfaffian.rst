.. _pfaffian:

Pfaffian pairing wave function
==============================
This section is a design document. The AGP/geminal singlet part (Milestone 1 below) is now
implemented in Pycasino — the :math:`\xi = 0` determinant :math:`\det[\Phi \,|\, \varphi]` with its
reader, ``Wfn`` integration and finite-difference tests; the Pfaffian proper (the triplet
:math:`\mu^\sigma` and inhomogeneous :math:`p_{klmI}` pairing terms, Milestone 2) is not yet.
It records the theory, the explicit functional form and the implementation plan; what is actually
available today is described in :ref:`geminal-in-pycasino`.

Motivation: nodal topology beyond backflow
------------------------------------------
As discussed in :ref:`backflow-nodes`, the backflow transformation is a homeomorphism of the
configuration space and cannot change the topology of the nodal surface. In fact the limitation
is stronger and applies to *any* wave function of the spin-factorized product form

.. math::

    \Psi(\mathbf{R}) = A(\mathbf{R})\,B(\mathbf{R})

where :math:`A` is antisymmetric in the spin-up electrons (and changes sign somewhere), :math:`B`
in the spin-down electrons — regardless of how :math:`A` and :math:`B` depend on *all* coordinates.
The sign of :math:`\Psi` is :math:`\operatorname{sign}(A)\operatorname{sign}(B)`, and a path from a
:math:`(+,+)` region to a :math:`(-,-)` region must cross :math:`A = 0`, where :math:`\Psi = 0`.
Both regions are non-empty (exchanging one up pair and one down pair maps one onto the other), so
:math:`\{\Psi > 0\}` is disconnected: **every product-form wave function has at least four nodal
pockets**, while the exact ground state generically has two [1]_.

This rules out, for the purpose of topological repair, not only the standard backflow of
:ref:`the backflow section <backflow>` but also its orbital-dependent generalizations [2]_ (a
different quasiparticle coordinate per orbital still leaves :math:`\Psi = D^\uparrow D^\downarrow`)
and even neural backflow in block-diagonal form: a single product of spin blocks is topologically
stuck, which is why such ansätze use sums of products or a full determinant mixing both spins.

Two explicit constructions break the factorization:

- a **multideterminant expansion** — a sum of at least two products (see the discussion of the
  :math:`2p^2` CSF for beryllium in :ref:`backflow-nodes`); the coefficients place the node and
  must be optimized in the presence of the Jastrow factor;
- a **pairing wave function** — a single antisymmetric object in which opposite-spin electrons are
  coupled inside the matrix elements. This is the geminal (AGP) [3]_ and, in full generality, the
  Pfaffian [4]_ [5]_ wave function described here. Bajdich et al. verified explicitly that the
  Pfaffian node of first-row atoms has the correct two-pocket topology, in contrast to the four
  pockets of the Hartree-Fock node [4]_.

The Pfaffian form
-----------------
The Pfaffian of an even-dimensional antisymmetric matrix :math:`M = -M^T` is defined by

.. math::

    \operatorname{pf}(M) = \frac{1}{2^n n!}\sum_{P} \operatorname{sign}(P)
    \prod_{k=1}^{n} M_{P(2k-1),P(2k)}, \qquad \operatorname{pf}(M)^2 = \det(M)

i.e. the antisymmetrized product of pairs — exactly what a BCS/RVB-type pairing state is. The
singlet-triplet-unpaired (STU) Pfaffian wave function [5]_ is

.. math::

    \Psi = e^{J(\mathbf{R})}\operatorname{pf}
    \begin{pmatrix}
    \xi^{\uparrow\uparrow} & \Phi^{\uparrow\downarrow} & \varphi^{\uparrow} \\
    -(\Phi^{\uparrow\downarrow})^T & \xi^{\downarrow\downarrow} & \varphi^{\downarrow} \\
    -(\varphi^{\uparrow})^T & -(\varphi^{\downarrow})^T & 0
    \end{pmatrix}

where the blocks are explicit functions of electron pairs:

- :math:`\Phi^{\uparrow\downarrow}_{ij} = \Phi(\mathbf{r}_i, \mathbf{r}_j)` — **singlet pairing**
  between up-electron :math:`i` and down-electron :math:`j`, spatially symmetric,
  :math:`\Phi(\mathbf{r}_i, \mathbf{r}_j) = \Phi(\mathbf{r}_j, \mathbf{r}_i)`;
- :math:`\xi^{\sigma\sigma}_{ij} = \xi^{\sigma}(\mathbf{r}_i, \mathbf{r}_j)` — **triplet pairing**
  within one spin channel, spatially antisymmetric;
- :math:`\varphi_{in} = \phi_n(\mathbf{r}_i)` — **unpaired** one-electron orbitals (one column per
  unpaired orbital; for the :math:`{}^2P` boron atom, one). With :math:`N^\uparrow - N^\downarrow`
  unpaired orbitals the matrix dimension is :math:`2N^\uparrow`, which is even as required.

Limiting cases make the ansatz fully interpretable:

- :math:`\Phi = \sum_{n}^{occ} \phi_n(\mathbf{r}_i)\phi_n(\mathbf{r}_j)`, :math:`\xi = 0` — the
  Pfaffian reduces exactly to the Hartree-Fock determinant;
- :math:`\xi = 0`, general symmetric :math:`\Phi` — the AGP (geminal) wave function [3]_;
- general :math:`\Phi`, :math:`\xi` — the STU Pfaffian, which contains both as special cases.

Expanding the Pfaffian in powers of the occupied-virtual part of the pairing shows what it is in
configuration-interaction terms: a sum over *all* double, quadruple, ... excitations whose
coefficients are factorized through the pairing matrix — a low-rank CI with a number of parameters
polynomial in the system size instead of the (non-size-consistent) exponential growth of an
explicit determinant list. In the language of :ref:`backflow-nodes`, the pairing term supplies the
:math:`-\varepsilon g` contribution that does not vanish on the intersection manifold
:math:`\{D^\uparrow = D^\downarrow = 0\}` and performs the topological surgery, with the
"coefficient" built into the antisymmetric form itself.

Pairing functions in explicit form
----------------------------------
The pairing functions are parametrized in the same explicit, term-by-term spirit as the
:math:`\eta`, :math:`\mu`, :math:`\Phi/\Theta` backflow functions.

**Molecular-orbital expansion.** With :math:`\{\phi_n\}` the one-electron orbitals of the existing
Slater machinery (including cusp correction),

.. math::

    \Phi(\mathbf{r}_i, \mathbf{r}_j) = \sum_{n,m} \lambda_{nm}\,\phi_n(\mathbf{r}_i)\phi_m(\mathbf{r}_j),
    \qquad \lambda_{nm} = \lambda_{mn}

.. math::

    \xi^{\sigma}(\mathbf{r}_i, \mathbf{r}_j) = \sum_{n,m} \mu^{\sigma}_{nm}
    \left[\phi_n(\mathbf{r}_i)\phi_m(\mathbf{r}_j) - \phi_m(\mathbf{r}_i)\phi_n(\mathbf{r}_j)\right],
    \qquad \mu^{\sigma}_{nm} = -\mu^{\sigma}_{mn}

Each :math:`\lambda_{nm}` is the amplitude of the pairing channel :math:`(n, m)`; the occupied
diagonal block :math:`\lambda_{nn} = 1` reproduces Hartree-Fock and the occupied-virtual block
carries the static correlation (for beryllium, :math:`\lambda_{2s,2s} = 1` plus a single optimizable
:math:`\lambda_{2p_\alpha,2p_\alpha}` common to :math:`\alpha = x, y, z` plays the role of the
:math:`2s^2 \to 2p^2` CSF coefficient). For an atomic :math:`S` state the pairing must be a scalar:
:math:`\lambda` couples orbitals within the same irreducible representation, with equal coefficients
across the degenerate components. Real orbitals and real symmetric :math:`\lambda` keep the wave
function real.

**Inhomogeneous short-range pairing.** The genuinely new explicit term — the analogue of the
:math:`\Phi` backflow term living *inside* the matrix element instead of in the coordinate
displacement, which is what allows it to affect the nodal topology:

.. math::

    \Phi(\mathbf{r}_i, \mathbf{r}_j) \mathrel{+}= \sum_{I=1}^{N_I}
    f(r_{iI}; L_I)f(r_{jI}; L_I)
    \sum_{k,l,m} p_{klmI}\, r_{iI}^k r_{jI}^l r_{ij}^m

with the same smooth cutoff :math:`f(r; L) = (1 - r/L)^C H(L - r)` as the backflow terms. The
constraints mirror the existing ones:

- symmetry of the singlet block: :math:`p_{klmI} = p_{lkmI}`;
- no electron-electron cusp in the antisymmetric part (Kato cusps are the Jastrow's job, and an
  :math:`r_{ij}`-linear term would produce a divergent local energy at coalescence):
  :math:`p_{kl1I} = 0`;
- smoothness at all-electron nuclei (as for the :math:`\mu` term): the :math:`r_{iI}`-expansion
  must start at second order, :math:`p_{0lmI}` and :math:`p_{1lmI}` constrained so that the entry
  behaves as :math:`O(r_{iI}^2)`;
- the cutoff makes the term size-consistent: distant subsystems decouple and the Pfaffian
  factorizes into the product of subsystem Pfaffians.

**Spurious high-energy configurations and their cancellation.** The pair expansion of a single
AGP determinant generates *all* products of pairing channels. A geminal
:math:`\Phi_2 = \phi_{1s}\phi_{1s} + \varepsilon \sum_p \phi_p \phi_p` describing the
:math:`2s^2 \to 2p^2` channel of beryllium therefore also contains the pair products
:math:`(p_i^2, p_j^2)` at amplitude :math:`\varepsilon^2` — configurations with an *empty*
:math:`1s` core, some :math:`\sim 10` hartree above the ground state. The resulting variational
penalty is quartic in the channel amplitude, :math:`\Delta E \approx 3\varepsilon^4 (E_j - E_0)`,
and at the optimal :math:`\varepsilon \approx -0.19` of beryllium it destroys most of the CSF
gain (tens of mHa). Because the determinant expansion of a sum of geminals is linear in the
geminals, such junk can be cancelled *exactly* by subtracting a geminal containing only the
offending channels:

.. math::

    \Psi = \det[\Phi_{HF}] + \det[\Phi_{1s} + \varepsilon \Phi_{2p}]
    - \det[\varepsilon \Phi_{2p}] \equiv D_{HF} + \varepsilon \sum_p D_p

which reproduces the pure CSF expansion with no spurious terms (verified to machine precision
against the multideterminant form). During optimization the amplitudes of the second and third
geminals must be constrained equal, or the cancellation degrades.

Evaluation
----------
Everything needed for VMC/DMC has the same computational shape as the Slater determinant:

- **Value.** :math:`\operatorname{pf}(M)` by skew-symmetric elimination (Parlett-Reid),
  :math:`O(N^3)`, with :math:`\operatorname{pf}(M)^2 = \det(M)` as a built-in sanity check.
- **Derivatives.** :math:`\nabla \ln \operatorname{pf}(M) = \tfrac{1}{2}\operatorname{tr}
  (M^{-1}\nabla M)`, and analogous second-derivative formulas via :math:`M^{-1}`; the inverse of a
  skew-symmetric matrix is skew-symmetric. Gradient, Laplacian and parameter derivatives follow the
  same pattern as the Slater ``gradient`` / ``laplacian`` / ``hessian`` chain.
- **Updates.** Moving one electron changes one row and one column: a rank-2 update of
  :math:`M^{-1}` (Woodbury), :math:`O(N^2)` per move — the Pfaffian analogue of Sherman-Morrison.
- **Backflow composition.** All entries are evaluated at the transformed coordinates
  :math:`\mathbf{x}_i`; the chain rule through the backflow Jacobian is identical to the Slater
  case, so the existing composition in ``casino/wfn.py`` carries over unchanged. Backflow refines
  the shape of the (now topologically correct) Pfaffian node.

.. _geminal-in-pycasino:

The geminal wave function in Pycasino
-------------------------------------
The implemented subset is the :math:`\xi = 0` limit above, in the multi-geminal (MAGP) form: the
Slater part is replaced by a sum of :math:`N^\uparrow \times N^\uparrow` determinants

.. math::

    \Psi_S = \sum_n c_n \det M_n, \qquad
    M_n[i, j] = \sum_{p,q} \phi_p(\mathbf{r}_i^\uparrow)\, g^n_{pq}\, \phi_q(\mathbf{r}_j^\downarrow),
    \qquad
    M_n[i, N^\downarrow + k] = \sum_p \phi_p(\mathbf{r}_i^\uparrow)\, u^n_{pk}

where the first :math:`N^\downarrow` columns are the singlet pairing columns and the remaining
:math:`N^\uparrow - N^\downarrow` are the unpaired-orbital columns of the open shell. The orbital
pool :math:`\{\phi_p\}` is the one from ``gwfn.data`` / ``stowfn.data``, evaluated by the same
machinery as the Slater determinant, so no Pfaffian kernel is needed at this stage.

The geminal wave function is selected by the ``psi_s : geminal`` input keyword and is mutually
exclusive with a multideterminant expansion; ``opt_geminal`` marks its parameters for optimization.
The parameters :math:`c_n`, :math:`g^n_{pq}` and :math:`u^n_{pk}` are read from the top-level
``GEMINAL`` block of ``parameters.casl``, with a per-element ``fixed``/``optimizable`` flag::

    GEMINAL:
      Default g optimizability: fixed
      Default c optimizability: fixed
      Geminal 1:
        Parameters:
          c: [ 1.0, fixed ]
          g_1,1: [ 1.0, fixed ]
          g_2,2: [ 1.0, fixed ]
          u_3,1: [ 1.0, fixed ]
          u_4,2: [ 1.0, fixed ]
          u_5,3: [ 1.0, fixed ]

Only the nonzero elements are listed; indices are one-based, :math:`p` runs over the orbital pool
and :math:`k` over the unpaired columns. When the block is absent, the Hartree-Fock default is
built from the occupation of the wave-function file — a single geminal with :math:`g = \mathbb{1}`
on the doubly occupied orbitals and one unpaired column per singly occupied orbital — which
reproduces the single Slater determinant exactly; this equality of value, gradient and Laplacian to
machine precision is the mandatory self-check of the implementation.

The geminal part of the wave function is represented by the :class:`casino.Geminal` class, which is
a drop-in replacement for :class:`casino.Slater` and is initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.geminal import Geminal

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    geminal = Geminal(config)

It has the following methods, all taking the electron-nuclei vectors ``n_vectors`` of shape
:math:`(N_{atom}, N_e, 3)`:

.. list-table::
   :widths: 30 40 30
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - ``value``
     - :math:`\Psi_S = \sum_n c_n \det M_n`
     - scalar
   * - ``gradient``
     - :math:`\nabla \Psi_S / \Psi_S`
     - :math:`(3N_e,)`
   * - ``laplacian``
     - :math:`\Delta \Psi_S / \Psi_S`
     - scalar
   * - ``pool_matrix``
     - :math:`\phi_p(\mathbf{r}_i^\uparrow)`, :math:`\phi_p(\mathbf{r}_i^\downarrow)`
     - :math:`(N_{orb}, N^\uparrow)`, :math:`(N_{orb}, N^\downarrow)`
   * - ``geminal_matrix``
     - :math:`M_n` from the orbital pool
     - :math:`(N^\uparrow, N^\uparrow)`

The determinant is recomputed on every move; Sherman-Morrison updates, cusp correction and the
parameter-optimization interface are not yet in place (see the implementation plan below).

Optimization strategy
---------------------
The pairing parameters place the nodal surface, and fixed-node DMC does not improve the node:
everything that determines it — :math:`\lambda`, :math:`\mu^\sigma`, :math:`p_{klmI}`, the cutoffs
— must therefore be optimized at the VMC level, exactly like CSF coefficients in a
multideterminant expansion.

Unlike CSF coefficients, which enter :math:`\Psi` linearly (and which the linear method recovers
in a single diagonalization), the pairing parameters are nonlinear: expanding the Pfaffian
multiplies each :math:`\lambda_{nm}` across all pair products. They are, however, as cheap to
differentiate as backflow parameters,

.. math::

    \frac{\partial \ln \operatorname{pf}(M)}{\partial \lambda_{nm}} =
    \frac{1}{2}\operatorname{tr}\left(M^{-1}\frac{\partial M}{\partial \lambda_{nm}}\right)

where :math:`\partial M / \partial \lambda_{nm}` is sparse (a single
:math:`\phi_n(\mathbf{r}_i)\phi_m(\mathbf{r}_j)` block pattern), so the Pfaffian plugs into the
standard varmin/emin machinery through the same first-derivative interface
(``value_parameters_d1``) as the existing terms, without changes to the optimizer itself.

**Staging.** The recommended order, mirroring the backflow discipline:

1. Jastrow alone, with :math:`\lambda = \mathbb{1}_{occ}` frozen — this is still the Hartree-Fock
   node;
2. the occupied-virtual block of :math:`\lambda`, initialized from the pair-excitation (seniority-0)
   CI coefficients of a small CASSCF when available, :math:`\lambda_v \approx c_v / c_0` — this is
   the step that performs the topological repair;
3. the triplet matrices :math:`\mu^\sigma` and the inhomogeneous term :math:`p_{klmI}` with its
   cutoffs (the most nonlinear parameters last);
4. a final joint cycle of all parameters, including the Jastrow.

**Redundancies.** The occupied-occupied block of :math:`\lambda` is frozen at :math:`\mathbb{1}`:
it is equivalent to orbital rotations plus overall normalization, and leaving it free creates flat
directions that stall the optimization. Symmetry ties (equal :math:`\lambda` across degenerate
components, e.g. :math:`2p_x, 2p_y, 2p_z` of an atom) are imposed through the parameter mask, as
for the spin-dependent backflow coefficients.

**Convergence criterion.** The Pfaffian pays off in the node, not in the VMC energy: on top of a
good Jastrow the VMC gain can be modest while the DMC gain is large. Energy minimization is
preferred over variance minimization for the pairing parameters (the energy is sensitive to the
node through the local-energy behavior near it), and the final quality check is the DMC energy, as
in the validation step below.

Implementation plan
-------------------
The Pfaffian class must be a drop-in replacement for the Slater part: the ``Wfn`` class interacts
with it only through ``value``, ``gradient``, ``laplacian``, ``hessian``, ``tressian`` /
``tressian_dot``, the parameter interface (``get_parameters``, ``set_parameters``,
``*_parameters_d1``, parameter masks) and the profile of ``n_vectors`` arguments.

The work is split into two milestones. Milestone 1 covers exactly the subset CASINO also supports
— the AGP/geminal singlet part in CASINO's own format — so that every step can be optimized by
*both* programs and the results compared; milestone 2 adds the Pfaffian-specific extensions that
exist in Pycasino only.

**Milestone 1 — AGP in CASINO format, cross-validated.** Steps 2 and 3 are implemented
(``casino/geminal.py``, ``casino/readers/geminal.py``, ``casino/tests/test_geminal.py``); step 4 is
partially done — the ``psi_s : geminal`` / ``opt_geminal`` input keywords and the ``Wfn`` value /
gradient / Laplacian composition are in place, the parameter-optimization interface and the Be/Ne
tests are not yet; steps 1 and 5 (generator, cross-validation) are open.

1. **Generator** (``molden2qmc``). From an ORCA (or other supported code) calculation, produce
   ``gwfn.data`` as now *plus* the ``GEMINAL`` block of ``parameters.casl`` in CASINO format
   (sparse ``g_n,m`` entries with per-element ``fixed``/``optimizable`` flags; the block references
   the orbital pool of ``gwfn.data``, so both files are needed). No quantum chemistry package
   generates this block, and ``molden2qmc`` already parses everything required — the orbital
   occupations of the ``[MO]`` section: integer occupations give the Hartree-Fock default
   (:math:`g = 1` on doubly-occupied orbitals, singly-occupied ones as unpaired columns, zeros with
   ``optimizable`` flags on a pool of virtuals), while fractional occupations of CASSCF natural
   orbitals seed the diagonal directly, :math:`\lambda_n = \pm\sqrt{n_n/2}` (minus for weakly
   occupied correlating orbitals; Molden files carry no CI vectors, so signs and off-diagonal
   elements are left to the VMC optimization or to an Orca-output parser).
2. **Reader** (``casino/readers/geminal.py``). Parse the top-level ``GEMINAL`` block of
   ``parameters.casl`` (``c``, ``g_n,m``, ``u_n,k`` with per-element ``fixed``/``optimizable``
   flags). When the block is absent, build the Hartree-Fock default from the occupation in the
   wave-function file (:math:`g = 1` on the doubly-occupied orbitals, unpaired columns for the
   singly-occupied ones); ``write`` regenerates the block after each optimization cycle.
3. **Geminal evaluation.** With :math:`\xi = 0` the Pfaffian reduces to an ordinary
   :math:`N^\uparrow \times N^\uparrow` determinant :math:`\det[\Phi \,|\, \varphi]` — rows indexed
   by the up electrons, :math:`N^\downarrow` pairing columns
   :math:`\Phi(\mathbf{r}_i^\uparrow, \mathbf{r}_j^\downarrow)` and
   :math:`N^\uparrow - N^\downarrow` unpaired-orbital columns — so no Pfaffian kernel is needed at
   this stage and the same Slater-style value/gradient/Laplacian AO evaluation is reused (the
   determinant is recomputed per move for now; Sherman-Morrison updates and cusp correction are
   deferred to a later stage). Mandatory self-check, now satisfied: with
   :math:`\lambda = \mathbb{1}_{occ}` the class reproduces the Slater single-determinant value,
   gradient and Laplacian to machine precision (``test_geminal.py``).
4. **Integration and optimization** (``casino/wfn.py``, ``casino/readers/input.py``). Input
   keywords following CASINO (``psi_s : geminal``, ``opt_geminal``); Jastrow and backflow
   composition unchanged; MDET and pairing mutually exclusive. Expose :math:`\lambda` through the
   standard varmin/emin parameter interface following the optimization strategy above.
   Finite-difference tests (``casino/tests/test_geminal.py``) for the gradient, Laplacian and
   parameter derivatives, on He (sanity) and Be/Ne (nontrivial :math:`l > 0` and
   :math:`n_{eu} \geq 2` paths).
5. **Cross-validation against CASINO.** Optimize the same wave function with both programs on the
   ``examples/geminal`` systems (He, Be, N, Ne, ...): compare VMC energies, DMC energies and the
   optimized ``g`` matrices. Beryllium is the acid test of the nodal physics: the optimization must
   develop :math:`\lambda_{2p,2p} \neq 0` and move the DMC energy from the single-determinant
   :math:`-14.6572` towards the 2-CSF quality :math:`\approx -14.6672` (exact :math:`-14.66736`).

**Milestone 2 — Pfaffian extensions (Pycasino only).**

6. **Pfaffian kernel.** ``pfaffian(M)`` and skew-symmetric inverse in nopython Numba, plus rank-2
   update. Unit tests: :math:`\operatorname{pf}(M)^2 = \det(M)` on random skew matrices, update
   against recomputation; the AGP limit must reproduce the milestone-1 results.
7. **Triplet and inhomogeneous terms.** The :math:`\mu^\sigma` matrices and the :math:`p_{klmI}`
   polynomial with its orders, cutoffs and AE-constraint flags, stored in extension sections of
   ``parameters.casl`` that CASINO ignores; constraints as in the pairing section above;
   hessian/tressian derivatives for energy minimization.
8. **Validation.** The B, C, N series where the CSF route saturates slowly (see [4]_ [5]_ for
   reference Pfaffian energies). Optionally, a nodal-pocket connectivity check (random-walk sign
   diagnostic of Bressanini) as an example script.

Open questions
--------------
- Parametrization redundancy: :math:`\lambda` mixes with orbital rotations and with the Jastrow;
  needs the same MAD/emin discipline as backflow optimization.
- Number of pairing orbitals: truncating the virtual space controls both cost and the variational
  freedom; natural orbitals of a small CASSCF are the natural starting set.
- Hessian/tressian volume: emin with backflow requires third derivatives
  (``tressian_dot``); deriving and testing these for the Pfaffian is the largest single chunk of
  work and can be deferred (varmin needs only first derivatives).
- Spin-contamination: the STU Pfaffian is not an :math:`\hat{S}^2` eigenfunction in general;
  acceptable for energy-oriented use (as with UHF-based nodes), worth documenting.

References
----------

.. [1] L. Mitas,
   *Structure of fermion nodes and nodal cells*,
   Phys. Rev. Lett. **96**, 240402 (2006); arXiv:cond-mat/0605550.

.. [2] M. Holzmann and S. Moroni,
   *Orbital-dependent backflow wave functions for real-space quantum Monte Carlo*,
   Phys. Rev. B **99**, 085121 (2019).

.. [3] M. Casula and S. Sorella,
   *Geminal wave functions with Jastrow correlation: A first application to atoms*,
   J. Chem. Phys. **119**, 6500 (2003).

.. [4] M. Bajdich, L. Mitas, G. Drobný, L. K. Wagner, and K. E. Schmidt,
   *Pfaffian Pairing Wave Functions in Electronic-Structure Quantum Monte Carlo Simulations*,
   Phys. Rev. Lett. **96**, 130201 (2006).

.. [5] M. Bajdich, L. Mitas, L. K. Wagner, and K. E. Schmidt,
   *Pfaffian pairing and backflow wavefunctions for electronic structure quantum Monte Carlo methods*,
   Phys. Rev. B **77**, 115112 (2008).
