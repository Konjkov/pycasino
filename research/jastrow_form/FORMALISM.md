# Mathematical formalism

This document states precisely the definitions behind `REPORT.md` and the scripts:
- the CASINO terms;
- the window and the proposed forms, with their cusp conditions;
- the weights and error metrics;
- the mean-field χ;
- the separability models for f;
- the local-density hole;
- the scaling laws and the parameter counting.

## 0. Notation

- **Coordinates.** Electrons $i, j$ with spins $\sigma_i$; nuclei $I$ with charge $Z_I$ at $\mathbf R_I$.
  $\mathbf r_{ij} = \mathbf r_i - \mathbf r_j$, $\mathbf r_{iI} = \mathbf r_i - \mathbf R_I$, and
  $r = |\mathbf r|$.
- **Counts and truncation.** $N = N_\uparrow + N_\downarrow$ is the number of electrons. $C$ is the
  truncation order; $C = 3$ in every file used here.
- **Step function.** $\theta(x) = 1$ for $x > 0$ and $0$ otherwise.
- **Trial wave function.** Slater–Jastrow–backflow:

$$
\Psi(\mathbf R) = e^{J(\mathbf R)}\, D(\mathbf x_1,\dots,\mathbf x_N),\qquad
\mathbf x_i = \mathbf r_i + \boldsymbol\xi_i(\mathbf R).
$$

- **Jastrow factor:**

$$
J = \sum_{i<j} u_{\sigma_i\sigma_j}(r_{ij}) + \sum_{i,I} \chi_{I,\sigma_i}(r_{iI})
  + \sum_{i<j,\,I} f_{I,\sigma_i\sigma_j}(r_{iI}, r_{jI}, r_{ij}).
$$

- **Backflow displacement:**

$$
\boldsymbol\xi_i = \sum_{j\ne i} \eta_{\sigma_i\sigma_j}(r_{ij})\,\mathbf r_{ij}
 + \sum_I \mu_{I,\sigma_i}(r_{iI})\,\mathbf r_{iI}
 + \sum_{I}\sum_{j\ne i}\Big[\Phi_{I}(r_{iI}, r_{jI}, r_{ij})\,\mathbf r_{ij} + \Theta_{I}(r_{iI}, r_{jI}, r_{ij})\,\mathbf r_{iI}\Big].
$$

  The η and Φ/Θ contributions of an electron near an all-electron (AE) nucleus are multiplied by
  $g(r_{iI}) = (r/L_g)^2\,[6 - 8 r/L_g + 3 (r/L_g)^2]$ for $r < L_g$. Here $L_g = 1$ bohr.

- **Cusp values.** $\Gamma_{\sigma\sigma'} = 1/4$ for parallel spins and $1/2$ for antiparallel spins.

## 1. CASINO forms (as evaluated in `casino/jastrow.py`, `casino/backflow.py`)

### 1.1 Jastrow

$$
u_s(r) = \theta(L_u - r)\,(r - L_u)^C \sum_{k=0}^{N_u} \alpha_{k,s}\, r^k,
\qquad
\chi_{I,s}(r) = \theta(L_\chi - r)\,(r - L_\chi)^C \sum_{k=0}^{N_\chi} \beta_{k,s}\, r^k,
$$

$$
f_{I,s}(r_1, r_2, r_{12}) = \theta(L_f - r_1)\,\theta(L_f - r_2)\,(r_1 - L_f)^C (r_2 - L_f)^C
\sum_{l,m=0}^{N_{eN}}\sum_{n=0}^{N_{ee}} \gamma_{lmn,s}\, r_1^l r_2^m r_{12}^n,
\qquad \gamma_{lmn} = \gamma_{mln}.
$$

**Kato e-e cusp on u.** Write $u = (r - L)^C P(r)$. Then

$$
u'(0) = C(-L)^{C-1}\alpha_0 + (-L)^C \alpha_1 = \Gamma
\;\Longrightarrow\;
\alpha_1 = \frac{\Gamma}{(-L)^C} + \frac{C\,\alpha_0}{L}.
$$

**No-cusp condition on χ.** In all data `chi_cusp = 0`, because the Kato e-n cusp is carried by the
(cusp-corrected or Slater) orbitals. The condition is $\chi'(0) = 0$, i.e. $\beta_1 = C\beta_0/L$.

**Constraints on f** (linear in γ):
- the e-e no-cusp condition, $\partial_{r_{12}} f\,|_{r_{12}=0} = 0$;
- the e-n no-cusp condition, $\partial_{r_1} f\,|_{r_1=0} = 0$;
- in all data `no_dup_u = no_dup_chi = 0`, so f may contain functions of $r_{12}$ alone and of $r_1$ alone
  inside the cutoff.

### 1.2 Backflow

The cutoff is normalized, $(1 - r/L)^C$:

$$
\eta_s(r) = \theta(L_\eta - r)\Big(1 - \frac{r}{L_\eta}\Big)^C \sum_{k=0}^{N_\eta} c_{k,s} r^k,\qquad
\mu_{I}(r) = \theta(L_\mu - r)\Big(1 - \frac{r}{L_\mu}\Big)^C \sum_{k=0}^{N_\mu} d_{k} r^k,
$$

$$
\{\Phi,\Theta\}_{I,s} = \Big(1 - \frac{r_1}{L_\Phi}\Big)^C\Big(1 - \frac{r_2}{L_\Phi}\Big)^C
\sum_{k,l=0}^{N_{eN}}\sum_{m=0}^{N_{ee}} \{\varphi,\vartheta\}_{klm,s}\, r_1^k r_2^l r_{12}^m .
$$

Constraints:
- parallel η: $\eta'(0) = 0$, i.e. $c_1 = C c_0 / L$;
- μ at an AE nucleus: $d_0 = d_1 = 0$, so $\mu \propto r^2$ near the nucleus;
- μ at a PP nucleus: $d_1 = C d_0/L$, i.e. $\mu'(0) = 0$.

## 2. The window and the construction of new forms

Every proposed radial form is

$$
y(r) = F(r;\boldsymbol\theta)\; w_C\!\left(\frac{r}{L}\right),\qquad
w_C(x) = (1 - x)^C (1 + C x)\,\theta(1 - x).
$$

**Properties of the window.**
1. $w_C(0) = 1$ and $w_C'(x) = -C(C+1)\,x\,(1-x)^{C-1}$, so $w_C'(0) = 0$ and $w_C''(0) = -C(C+1)$.
2. Because of the factor $(1-x)^C$, $w_C^{(k)}(1) = 0$ for $k = 0,\dots,C-1$. The cut-off function is
   therefore $C-1$ times continuously differentiable at $r = L$, exactly like CASINO's $(r - L)^C$.
3. From 1: $y(0) = F(0)$ and $y'(0) = F'(0)$. **The cusp condition is a condition on F alone and does
   not involve L.** This is unlike the CASINO polynomial, where $\alpha_1$ depends on $L$.
4. $w_3(x_{1/2}) = 1/2$ at $x_{1/2} \approx 0.386$. When the window is the whole shape (form `window`),
   $r_{1/2} = 0.386\,L$.

### 2.1 u: condition $F'(0) = \Gamma$

| name | $F(r)$ | check of $F'(0)$ | parameters |
|---|---|---|---|
| exp | $-\Gamma b\, e^{-r/b}$ | $\Gamma$ | $b$ |
| pade | $-\Gamma b/(1 + r/b)$ | $\Gamma$; tail $-\Gamma b^2/r$ | $b$ |
| rpa | $-2\Gamma F\,(1 - e^{-r/F})/(r/F)$ | $(1-e^{-x})/x = 1 - x/2 + O(x^2)$ gives $\Gamma$; tail $-2\Gamma F^2/r$ | $F$ |
| exp_cusp | $[-A + (\Gamma - A/b)\,r]\,e^{-r/b}$ | $A/b + \Gamma - A/b = \Gamma$ | $A, b$ |
| exp2 | $-\Gamma b_1 e^{-r/b_1} - A_2 (1 + r/b_2) e^{-r/b_2}$ | the second term has zero slope | $b_1, A_2, b_2$ |

Interpretation of `exp`: the depth is fixed by the cusp, $u(0) = -\Gamma b$. The single length b is the
radius of the Coulomb correlation hole.

A zero-slope term $A(1+r/b)e^{-r/b}$ is used repeatedly below:

$$
\frac{d}{dr}\Big[(1+r/b)e^{-r/b}\Big] = -\frac{r}{b^2}e^{-r/b} \;\Rightarrow\; 0 \text{ at } r=0 .
$$

### 2.2 χ: condition $F'(0) = 0$

| name | $F(r)$ | parameters |
|---|---|---|
| window | $A$ | $A$ (shape given by $w_C$; $L$ is the width) |
| gauss | $A e^{-(r/a)^2}$ | $A, a$ |
| yukawa | $A (1 + r/a) e^{-r/a}$ | $A, a$ |
| lorentz | $A/(1 + (r/a)^2)$ | $A, a$ |
| sech | $A/\cosh(r/a)$ | $A, a$ |
| yukawa2 | $\sum_{p=1}^{2} A_p (1 + r/a_p) e^{-r/a_p}$ | $A_1, a_1, A_2, a_2$ |

### 2.3 η: parallel $F'(0) = 0$, antiparallel free

- **exp:** $F = A e^{-r/b}$ (antiparallel) or $A(1 + r/b)e^{-r/b}$ (parallel).
- **gauss:** $F = A e^{-(r/b)^2}$.
- **exp_osc:** $F = F_{\rm exp}\cos(kr)$. Since $\cos$ is even, the slope condition of $F_{\rm exp}$ is kept.

### 2.4 μ: AE $F = O(r^2)$, PP $F'(0) = 0$

- **shell:** $F = A (r/a)^2 e^{-r/a}$ (AE), with an extremum at $r = 2a$; or $A(1 + r/a)e^{-r/a}$ (PP).
- **ring:** $F = A\,\big[e^{-((r - r_0)/s)^2} + e^{-((r + r_0)/s)^2}\big]$. This is an even function of r,
  so $F'(0) = 0$. At an AE nucleus it is multiplied by $1 - e^{-(r/s)^2} = O(r^2)$.

## 3. Weights, error metrics, fitting

### 3.1 Densities

The HF density is $\rho(\mathbf r) = \sum_{\sigma}\sum_{k \in {\rm occ}_\sigma} |\phi_{k\sigma}(\mathbf r)|^2$.
For a multideterminant wave function the first determinant is used. Two densities are built from it.

**Spherical average around nucleus I**, with a Fibonacci quadrature of $M = 302$ directions:

$$
\bar\rho_I(r) = \frac{1}{4\pi}\oint \rho(\mathbf R_I + r\hat{\mathbf n})\,d\hat{\mathbf n}
\approx \frac1M\sum_{m=1}^{M} \rho(\mathbf R_I + r\hat{\mathbf n}_m),\qquad
\hat{\mathbf n}_m = (\sqrt{1-z_m^2}\cos\varphi_m,\ \sqrt{1-z_m^2}\sin\varphi_m,\ z_m),
$$

with $z_m = 1 - 2(m - \tfrac12)/M$ and $\varphi_m = \pi(1+\sqrt5)(m - \tfrac12)$.

**Independent-pair distance distribution**, estimated from Metropolis samples of $\rho/N$:

$$
P(r) = \frac{1}{N^2}\iint \rho(\mathbf x)\rho(\mathbf y)\,\delta(|\mathbf x - \mathbf y| - r)\,d\mathbf x\,d\mathbf y .
$$

This neglects the exchange-correlation hole, which is acceptable for a weight.

### 3.2 Weights

$$
w_{\rm en}(r) = 4\pi r^2 \bar\rho_I(r)\quad(\chi,\ \mu),\qquad
w_{\rm ee}(r) = P(r)\quad(u,\ \eta),
$$

each normalized to $\max w = 1$. For a set of several nuclei, $\bar\rho_I$ is averaged over the set.

### 3.3 Error metrics

A term has spin channels $s$ (only channels that occur in the system are kept). For a model $\hat y_s$ of
a profile $y_s$:

$$
\varepsilon[\hat y; y] = \left(\frac{\sum_s \int w\,(\hat y_s - y_s)^2\,dr}{\sum_s \int w\, y_s^2\,dr}\right)^{1/2},
\qquad
\varepsilon_d[\hat y; y] = \varepsilon[\hat y'; y'] .
$$

The integrals are sums over the grid $0 \le r \le 12$ bohr (321 points). Derivatives are taken by finite
differences.

**Noise floor.** $\varepsilon_{\rm noise} = \varepsilon[y^{(B)}; y^{(A)}]$ for two optimizations A and B of
the same system: CASINO emin vs PyCasino emin (same objective), or emin vs varmin.

### 3.4 Fit

The parameters $\boldsymbol\theta_s$ of all channels and one shared cutoff L are found by

$$
\min_{\{\boldsymbol\theta_s\},\,L}\ \sum_s \sum_k w(r_k)\,\big[F(r_k;\boldsymbol\theta_s)\,w_C(r_k/L) - y_s(r_k)\big]^2 ,
$$

with bounds: lengths $> 0.02$ bohr and $0.5 \le L \le 15$ bohr. The problem is solved by
trust-region least squares from several starting points.

### 3.5 Reduced-order polynomial baseline

The design columns are $\phi_k(r) = r^k(1 - r/L)^C$ for $k = 0..N$. Since
$\hat y'(0) = c_1 - C c_0 / L$, each constraint is eliminated exactly:

| condition | model |
|---|---|
| cusp Γ | $\hat y = \Gamma\phi_1 + c_0\,(\phi_0 + \tfrac{C}{L}\phi_1) + \sum_{k\ge2} c_k\phi_k$ |
| zero slope | the same with $\Gamma = 0$ |
| $O(r^2)$ (AE μ) | $\hat y = \sum_{k\ge2} c_k\phi_k$ |
| free | $\hat y = \sum_{k\ge0} c_k\phi_k$ |

For fixed L the coefficients follow from weighted linear least squares. L is found by a bounded 1D
minimization of ε.

## 4. The u–χ gauge freedom

Suppose all $r_{ij} < L_u$ and $r_{iI} < L_\chi$ for a single-nucleus system. The substitution

$$
u \to u + c,\qquad \chi \to \chi - \tfrac{N-1}{2}c
$$

changes J by $\binom N2 c - N\frac{N-1}{2}c = 0$. The value of u relative to χ is therefore fixed only by
configurations in which some distances exceed the cutoffs. Hence ε between two optimizations is dominated
by this offset, and $\varepsilon_d$ is the more meaningful measure. (With the cutoffs the substitution is
exact only in the interior region.)

## 5. Density-preserving χ (mean field of u)

For $|\Psi|^2 = |D|^2 e^{2J}$ with $J = \sum\chi + \sum u$, the one-electron density of spin σ is

$$
\rho_\sigma(\mathbf r) \propto \rho^{\rm HF}_\sigma(\mathbf r)\, e^{2\chi_\sigma(\mathbf r)}
\Big\langle \exp\Big(2\sum_{j\ne i} u_{\sigma\sigma_j}(|\mathbf r - \mathbf r_j|)\Big)\Big\rangle_{\mathbf r_i = \mathbf r}.
$$

Two approximations are made:
- replace the average of the exponential by the exponential of the average, i.e. keep the first cumulant;
- take the other electrons independent, with $\rho_{\sigma'} \approx (N_{\sigma'}/N)\rho$.

This gives

$$
\rho_\sigma \propto \rho^{\rm HF}_\sigma\, e^{2[\chi_\sigma + V_\sigma]},\qquad
V_\sigma(\mathbf r) = \sum_{\sigma'} \frac{N_{\sigma'} - \delta_{\sigma\sigma'}}{N}\int \rho(\mathbf r')\,u_{\sigma\sigma'}(|\mathbf r - \mathbf r'|)\,d\mathbf r'.
$$

The χ that leaves the HF density unchanged is therefore

$$
\chi_\sigma(r) = -V_\sigma(r) + \text{const}.
$$

For a spherical density, V is evaluated with 48-point Gauss–Legendre quadrature in $t$:

$$
V(r) = 2\pi\int_0^\infty r'^2 \rho(r')\int_{-1}^{1} u\!\left(\sqrt{r^2 + r'^2 - 2 r r' t}\right)dt\,dr' .
$$

For a spin-independent χ, $V = \sum_\sigma N_\sigma V_\sigma / \sum_\sigma N_\sigma$.

**Test.** The weighted regression $\chi \approx \alpha + \beta V$ on $r < L_\chi$. The value $\beta = -1$
means exact density preservation; $1 - R^2$ is the part of χ that changes the density.

**Proposed one-parameter form:**

$$
\chi(r) = \beta\,[V(r) - V(L)]\,w_C(r/L).
$$

V is computed once from the HF density and u. The form satisfies $\chi'(0) = 0$, since $V'(0) = 0$ for a
spherical ρ.

## 6. Separability of f, Φ, Θ

### 6.1 Sampling

Triplets are drawn from the independent-electron distribution around nucleus I, restricted to the cutoff:

$$
r_1, r_2 \sim \frac{4\pi r^2\bar\rho_I(r)\,\theta(L - r)}{\int_0^L 4\pi r^2\bar\rho_I},\qquad
\cos\vartheta \sim U(-1,1),\qquad
r_{12} = \sqrt{r_1^2 + r_2^2 - 2 r_1 r_2\cos\vartheta}.
$$

$2\times10^4$ samples are used. Sample averages are then weighted averages.

### 6.2 Models

Scaled variables: $x_a = r_a/L$, $s = (x_1 + x_2)/2$, $d = |x_1 - x_2|$. The cutoff factor is
$\kappa = (1 - x_1)^C (1 - x_2)^C$. Each model is $\hat y = \kappa\, M$, where $\mathcal P_D(\cdot)$ denotes
all monomials of total degree $\le D$:

| model | $M$ | coefficients |
|---|---|---|
| additive | $A(x_1) + A(x_2) + H(x_{12})$, degree 4 | 9 |
| $(s, r_{12})$ | $\mathcal P_4(s, x_{12})$ | 15 |
| $(d, r_{12})$ | $\mathcal P_4(d, x_{12})$ | 15 |
| $(s, d)$ | $\mathcal P_4(s, d)$ | 15 |
| $(s, d^2, r_{12})$ | $\mathcal P_3(s, d^2, x_{12})$ (all symmetric cubics) | 20 |
| rank 1 | $g(x_1)\,g(x_2)\,h(x_{12})$, g and h cubic | 10 |

The quality of a model is

$$
R^2 = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}.
$$

The rank-1 model is fitted by alternating least squares on the symmetrized sample
$\{(r_1,r_2,r_{12})\}\cup\{(r_2,r_1,r_{12})\}$. With two of $g_1, g_2, h$ fixed, the third is a linear
problem; symmetrization makes $g_1 = g_2$ at convergence.

## 7. Local-density correlation hole

### 7.1 Slices

For $r_1 = r_2 = R$, the antiparallel pair function including f is

$$
u_{\rm eff}(r_{12}; R) = u_{\uparrow\downarrow}(r_{12}) + f(R, R, r_{12}),\qquad 0 \le r_{12} \le 2R .
$$

The e-e no-cusp constraint on f keeps $\partial_{r_{12}} u_{\rm eff}|_0 = \Gamma$ for every R. Each slice is
fitted by

$$
u_{\rm eff}(r_{12}; R) \approx c(R) - \Gamma\, b(R)\, e^{-r_{12}/b(R)} .
$$

Only slices with $0.06 < b(R) < R$ are kept, i.e. where the hole fits inside the slice.

### 7.2 Universal law

$b(R)$ is compared with the local Wigner–Seitz radius

$$
r_s(R) = \left(\frac{3}{4\pi\bar\rho(R)}\right)^{1/3}.
$$

The pooled fit over all atoms, in log space, gives

$$
b(r_s) = \frac{B\, r_s}{r_s + c},\qquad B = 3.24,\ c = 2.71 .
$$

Limits: $b \to (B/c)\,r_s = 1.20\,r_s$ for $r_s \ll c$ (high density, as in the homogeneous gas), and
$b \to B$ for $r_s \gg c$ (valence saturation).

### 7.3 Proposed three-body form (replaces u + f)

$$
u(r_{ij}; r_{iI}, r_{jI}) = -\Gamma\,\bar b\, e^{-r_{ij}/\bar b}\; w_C(r_{ij}/L),\qquad
\bar b = b\big(r_s(\bar\rho(\bar r))\big),\qquad
\bar r = \sqrt{\tfrac12 (r_{iI}^2 + r_{jI}^2)} .
$$

Both cusp conditions hold by construction:
- **e-e cusp.** $\bar b$ does not depend on $r_{ij}$, so $\partial_{r_{ij}} u|_{r_{ij}=0} = \Gamma$ holds
  exactly for any density dependence.
- **e-n no-cusp.** $\partial \bar r/\partial r_{iI} = r_{iI}/(2\bar r) \to 0$ as $r_{iI} \to 0$ (for
  $r_{jI} > 0$), so $\partial_{r_{iI}} u|_{r_{iI}=0} = 0$ and the e-n cusp of the orbitals is not
  disturbed. This is why the rms mean is used and not the arithmetic mean $(r_{iI} + r_{jI})/2$.

With several nuclei, $\bar\rho$ can be the molecular density at the pair's position, or a sum of atomic
contributions.

The form has two universal parameters (B, c), or one scale per species. It is a proposal: only its
agreement with the CASINO slices is tested here (R²(log) = 0.92).

## 8. Scaling laws

Model-free descriptors are read off the CASINO profiles:
- u: $b_0 = -u(0)/\Gamma$ (the exponential hole radius with the same depth);
- χ: $\chi(0)$ and $r_{1/2}$ defined by $\chi(r_{1/2}) = \chi(0)/2$;
- η: $\eta(0)$ and its half-width;
- μ: the position $r_{\rm ext}$ of the maximum of $w\,|\mu|$;
- the radius of the outermost shell, $r_{\rm out}$ = the position of the outermost maximum of
  $4\pi r^2\bar\rho$ (larger than 5 % of the global maximum).

Two kinds of fit are used:

$$
\text{power law: } \ln y = \ln c + \alpha \ln Z \quad(\text{least squares in log–log});\qquad
\text{shell law: } y = k\, r_{\rm out},\quad k = \frac{\sum r_{\rm out}\, y}{\sum r_{\rm out}^2}.
$$

**Atom → molecule transfer.** The window form $\chi = A\,w_C(r/L)$ uses $A(Z) = 0.903\,Z^{0.582}$ and
$L(Z) = 6.15\,Z^{-0.160}$, fitted over the AE atoms. It is compared with the molecular χ via
$\varepsilon$ and $\varepsilon_d$ (§3.3).

## 9. Anisotropy in molecules

Points on the bond from nucleus a to nucleus b: $\mathbf P(t) = \mathbf R_a + t(\mathbf R_b - \mathbf R_a)$.
A second electron is placed at $\mathbf P + \delta\hat{\mathbf n}_m$, with $\delta = 1$ bohr and 50 Fibonacci
directions. The ratios are

$$
\rho_f(t) = \frac{\big\langle [\sum_I f_I(|\mathbf P - \mathbf R_I|, |\mathbf P + \delta\hat{\mathbf n} - \mathbf R_I|, \delta)]^2\big\rangle_{\hat{\mathbf n}}^{1/2}}{|u_{\uparrow\downarrow}(\delta)|},
$$

$$
\rho_\Phi(t) = \frac{\big\langle |\sum_I \Phi_I\,(\mathbf P - \mathbf P') + \Theta_I\,(\mathbf P - \mathbf R_I)|^2\big\rangle_{\hat{\mathbf n}}^{1/2}}
{\big(\int P(r)\,[\eta_{\uparrow\downarrow}(r)\,r]^2 dr / \int P\big)^{1/2}},
\qquad \mathbf P' = \mathbf P + \delta\hat{\mathbf n}.
$$

A ratio $\ll 1$ means that radial one-body terms plus pair terms are enough in that region.

## 10. Symbolic regression

Profiles are pooled after scaling each one by quantities read off itself:

| term | x | y |
|---|---|---|
| u | $r/b_0$ | $u/\lvert u(0)\rvert$ |
| χ | $r/r_{1/2}$ | $\chi/\chi(0)$ |
| η | $r/r_{1/2}$ | $\eta/\lvert\eta(0)\rvert$ |

Points are restricted to $r < 0.7L$ and $w > 0.02$.

gplearn minimizes the weighted MSE $+\ p\cdot\ell$, where $\ell$ is the program length and
$p \in \{10^{-2}, 10^{-3}, 10^{-4}\}$. The primitives are $\{+,-,\times,\div,\exp\}$ and constants lie in
$[-2, 2]$.

Reference closed forms have no free parameter on the scaled axes:
- $-e^{-x}$ for u;
- $e^{-\ln 2\, x^2}$ for χ, which has half value at $x = 1$.

## 11. Parameter counting

**CASINO.** The number of `! alpha_ / beta_ / gamma_ / c_ / mu_ / phi_ / theta_` lines in the
correlation.out file (the independent parameters after the constraints), plus one per cutoff.

**Proposed**, with $n_s$ the number of spin channels that occur in the system:

| term | minimal | accurate |
|---|---|---|
| u | $1 + \sum_s p_s$, with $p_s = 1$ (exp) or 3 (exp2, if $u_s(0) > 0$) | $1 + 3 n_s$ |
| χ | per set $1 + n_s$ (window) | per set $1 + 4 n_s$ (yukawa2) |
| f | per set $1 + 10\, n_s$ (rank 1) | per set $1 + 20\, n_s$ (symmetric cubic) |
| η | $1 + 2 n_s$ | $1 + 3 n_s$ |
| μ | per set 3 | per set 4 |
| Φ/Θ | unchanged | unchanged |

The local-density hole of §7.3 would replace u + f by 2 universal parameters per spin channel plus one
cutoff.
