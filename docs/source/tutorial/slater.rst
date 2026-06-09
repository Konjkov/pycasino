.. _slater:

Slater determinant
==================

The Slater determinant component of the wavefunction is implemented in the :class:`casino.Slater` class.
This class provides methods to compute the value, gradient, Laplacian, Hessian, and Tressian of a multi-determinant Slater wavefunction.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.slater import Slater

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    slater = Slater(config, cusp=None)

Summary of Methods
------------------

Slater class has a following methods:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - :ref:`value_matrix <value-matrix>`
     - :math:`A^\uparrow, A^\downarrow`
     - :math:`(N^\uparrow_e, MO^\uparrow), (N^\downarrow_e, MO^\downarrow)`
   * - :ref:`gradient_matrix <gradient-matrix>`
     - :math:`G^\uparrow, G^\downarrow`
     - :math:`(N^\uparrow_e, MO^\uparrow, 3), (N^\downarrow_e, MO^\downarrow, 3)`
   * - :ref:`laplacian_matrix <laplacian-matrix>`
     - :math:`L^\uparrow, L^\downarrow`
     - :math:`(N^\uparrow_e, MO^\uparrow), (N^\downarrow_e, MO^\downarrow)`
   * - :ref:`hessian_matrix <hessian-matrix>`
     - :math:`H^\uparrow, H^\downarrow`
     - :math:`(N^\uparrow_e, MO^\uparrow, 3, 3), (N^\downarrow_e, MO^\downarrow, 3, 3)`
   * - :ref:`tressian_matrix <tressian-matrix>`
     - :math:`T^\uparrow, T^\downarrow`
     - :math:`(N^\uparrow_e, MO^\uparrow, 3, 3, 3), (N^\downarrow_e, MO^\downarrow, 3, 3, 3)`
   * - :ref:`value <value>`
     - :math:`\Psi(r)`
     - :math:`scalar`
   * - :ref:`gradient <gradient>`
     - :math:`\nabla \Psi(r)/\Psi(r)`
     - :math:`(3N_e,)`
   * - :ref:`laplacian <laplacian>`
     - :math:`\Delta \Psi(r)/\Psi(r)`
     - :math:`scalar`
   * - :ref:`hessian <hessian>`
     - :math:`\nabla^2 \Psi(r)/\Psi(r)`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`tressian <tressian>`
     - :math:`\nabla^3 \Psi(r)/\Psi(r)`
     - :math:`(3N_e, 3N_e, 3N_e)`

.. _value-matrix:

value matrix
------------

In quantum chemistry, molecular orbitals (MOs) are normally expanded in a set of atom-centered basis functions, or localized atomic orbitals (AOs):

.. math::

    \phi_p(\mathbf{r}) = \sum_{\alpha}c_{\alpha p}\chi_\alpha(\mathbf{r}-\mathbf{R}_\alpha)

where :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons, :math:`\mathbf{R}_\alpha` denotes the atomic
position center of basis function :math:`\chi_\alpha`, and the expansion coefficients :math:`c_{\alpha p}` are known as molecular orbital (MO) coefficients,
also to avoid overflows and underflows a normalization coefficient is multiplied:

.. math::

    A_{ip} = \frac{1}{\sqrt[2N]{N!}} \phi_p(r_i)


In multi-determinant case :math:`p` indexes should includes an occupied plus virtual MOs to span for excited states. Therefore it is reasonable to
define electrons :math:`\times` (occupied + virtual MOs) matrix :math:`\mathcal{A}`.
For a system described by a spin-independent Hamiltonian, the spatial and spin degrees of freedom are separable and we can split :math:`\mathcal{A}_{ip}`
into two: :math:`\mathcal{A}^\uparrow_{ip}` for spin-up and :math:`\mathcal{A}^\downarrow_{ip}` for spin-down electrons.
For certain electron coordinates, the values of these matrices can be obtained with :py:meth:`casino.Slater.value_matrix` method::

    import numpy as np

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    atom_positions = config.wfn.atom_positions
    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
    A_up, A_down = slater.value_matrix(n_vectors)

.. _inverse-matrix:

the inverse matrix will be needed to calculate the gradient, laplacian, hesian and tressian::

    inv_A_up = np.linalg.inv(A_up)
    inv_A_down = np.linalg.inv(A_down)

.. _gradient-matrix:

gradient matrix
---------------

Consider the gradient operator for :math:`i`-th electron:

.. math::

    \nabla_{e_i} = \left[\frac{\partial}{\partial{x_i}}, \frac{\partial}{\partial{y_i}}, \frac{\partial}{\partial{z_i}}\right]

It is easy to check that:

.. math::

    \nabla_{e_i} A_{jp} = 0 \quad \text{if} \quad i \neq j


hence all non-zero values compose the matrix of vectors: :math:`(x, y, z)` indexed by :math:`a \in (x, y, z)`:

.. math::

    G_{ipa} = \nabla_{e_i} A_{ip}

In multi-determinant case :math:`p` indexes should includes an occupied plus virtual MOs to span for excited states. Therefore it is reasonable to
define electrons :math:`\times` (occupied + virtual MOs) matrix :math:`\mathcal{G}_{ip}`.
For a system described by a spin-independent Hamiltonian, the spatial and spin degrees of freedom are separable and we can split :math:`\mathcal{G}_{ip}`
into two: :math:`\mathcal{G}^\uparrow_{ip}` for spin-up and :math:`\mathcal{G}^\downarrow_{ip}` for spin-down electrons.
For certain electron coordinates, the values of these matrices can be obtained with :py:meth:`casino.Slater.gradient_matrix` method::

    G_up, G_down = slater.gradient_matrix(n_vectors)

.. _laplacian-matrix:

laplacian matrix
----------------

Consider the laplacian operator for :math:`i`-th electron:

.. math::

    \Delta_{e_i} = \frac{\partial^2}{\partial{x_i}^2} + \frac{\partial^2}{\partial{y_i}^2} + \frac{\partial^2}{\partial{z_i}^2}

It is easy to check that:

.. math::

    \Delta_{e_i} A_{jp} = 0 \quad \text{if} \quad i \neq j

hence all non-zero values compose the matrix of scalars:


.. math::

    L_{ip} = \Delta_{e_i} A_{ip}

In multi-determinant case :math:`p` indexes should includes an occupied plus virtual MOs to span for excited states. Therefore it is reasonable to
define electrons :math:`\times` (occupied + virtual MOs) matrix :math:`\mathcal{L}_{ip}`.
For a system described by a spin-independent Hamiltonian, the spatial and spin degrees of freedom are separable and we can split :math:`\mathcal{L}_{ip}`
into two: :math:`\mathcal{L}^\uparrow_{ip}` for spin-up and :math:`\mathcal{L}^\downarrow_{ip}` for spin-down electrons.
For certain electron coordinates, the values of these matrices can be obtained with :py:meth:`casino.Slater.laplacian_matrix` method::

    L_up, L_down = slater.laplacian_matrix(n_vectors)

.. _hessian-matrix:

hessian matrix
--------------

Consider the hessian operator for :math:`i`-th electron:

.. math::

    \nabla_{e_i} \otimes \nabla_{e_i}

It is easy to check that:

.. math::

    (\nabla_{e_i} \otimes \nabla_{e_i}) A_{jp} = 0 \quad \text{if} \quad i \neq j

hence all non-zero values compose the matrix of hessians: :math:`(x, y, z) \otimes (x, y, z)` indexed by :math:`a,b \in (x, y, z)`:

.. math::

    H_{ipab} = (\nabla_{e_i} \otimes \nabla_{e_i}) A_{ip}

In multi-determinant case :math:`p` indexes should includes an occupied plus virtual MOs to span for excited states. Therefore it is reasonable to
define electrons :math:`\times` (occupied + virtual MOs) matrix :math:`\mathcal{H}_{ip}`.
For a system described by a spin-independent Hamiltonian, the spatial and spin degrees of freedom are separable and we can split :math:`\mathcal{H}_{ip}`
into two: :math:`\mathcal{H}^\uparrow_{ip}` for spin-up and :math:`\mathcal{H}^\downarrow_{ip}` for spin-down electrons.
For certain electron coordinates, the values of these matrices can be obtained with :py:meth:`casino.Slater.hessian_matrix` method::

    H_up, H_down = slater.hessian_matrix(n_vectors)

.. _tressian-matrix:

tressian matrix
---------------

Consider the tressian operator for :math:`i`-th electron:

.. math::

    \nabla_{e_i} \otimes \nabla_{e_i} \otimes \nabla_{e_i}

It is easy to check that:

.. math::

    (\nabla_{e_i} \otimes \nabla_{e_i} \otimes \nabla_{e_i}) A_{jp} = 0 \quad \text{if} \quad i \neq j

hence all non-zero values compose the matrix of tressians: :math:`(x, y, z) \otimes (x, y, z) \otimes (x, y, z)` indexed by :math:`a,b,c \in (x, y, z)`:

.. math::

    T_{ipabc} = (\nabla_{e_i} \otimes \nabla_{e_i} \otimes \nabla_{e_i}) A_{ip}


In multi-determinant case :math:`p` indexes should includes an occupied plus virtual MOs to span for excited states. Therefore it is reasonable to
define electrons :math:`\times` (occupied + virtual MOs) matrix :math:`\mathcal{T}_{ip}`.
For a system described by a spin-independent Hamiltonian, the spatial and spin degrees of freedom are separable and we can split :math:`\mathcal{T}_{ip}`
into two: :math:`\mathcal{T}^\uparrow_{ip}` for spin-up and :math:`\mathcal{T}^\downarrow_{ip}` for spin-down electrons.
For certain electron coordinates, the values of these matrices can be obtained with :py:meth:`casino.Slater.tressian_matrix` method::

    T_up, T_down = slater.tressian_matrix(n_vectors)

.. _value:

value
-----

Consider contribution of single Slater determinant:

.. math::

    \psi(\mathbf{r}) = \det(A)

we can get the value of multideterminant wavefunction:

.. math::

    \Psi(\mathbf{r}) = \sum_n c_n \psi(\mathbf{r})_n

and  :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons.

For certain electron coordinates, the value can be obtained with casino.Slater.value() method::

    value = slater.value(n_vectors)

.. _gradient:

gradient
--------

Consider Slater determinant gradien by :math:`i`-th electron coordinates:

.. math::

    \frac{\nabla_{e_i} \psi(\mathbf{r})}{\phi(\mathbf{r})} = \left[
    tr\left(A^{-1}\frac{\partial{A}}{\partial{x_i}}\right),
    tr\left(A^{-1}\frac{\partial{A}}{\partial{y_i}}\right),
    tr\left(A^{-1}\frac{\partial{A}}{\partial{z_i}}\right)
    \right] = tr(A^{-1} \nabla_{e_i} A)

to express the trace through sum using equality:

.. math::

    tr(AB) = \sum_{ij} a_{ij}b_{ji} = {a_i}^j {b_j}^i

notice that the :math:`\nabla_{e_i} A` has the only one non-zero :math:`row_i(\nabla_{e_i} A) = row_i(G)`:

.. math::

    tr(A^{-1} \nabla_{e_i} A) = {(A^{-1})_i}^j {(\nabla_{e_i} A)_j}^{ia}

expand gradient vector over :math:`i`:

.. math::

    \frac{\nabla \psi(\mathbf{r})}{\phi(\mathbf{r})} = {(A^{-1})_i}^j G_{jia}

and get gradient of multideterminant wavefunction:

.. math::

    \nabla \Psi(\mathbf{r}) / \Phi(\mathbf{r}) = \sum_n c_n \nabla \psi(\mathbf{r})_n / \sum_n c_n \psi(\mathbf{r})_n

where :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons

For certain electron coordinates, the gradient vector can be obtained with casino.Slater.gradient() method::

    slater.gradient(n_vectors)

this is equivalent to (continues :ref:`from <inverse-matrix>`)::

    G_up, G_down = slater.gradient_matrix(n_vectors)
    tr_grad_u = np.einsum('ij,jia->ia', inv_A_up, G_up).reshape(neu * 3)
    tr_grad_d = np.einsum('ij,jia->ia', inv_A_down, G_down).reshape(ned * 3)
    np.concatenate((tr_grad_u, tr_grad_d))

.. _laplacian:

laplacian
---------

Consider Slater determinant laplacian by :math:`i`-th electron coordinates:


.. math::

    \frac{\Delta_{e_i} \phi(\mathbf{r})}{\phi(\mathbf{r})} =
    tr\left(A^{-1}\frac{\partial^2{A}}{\partial{x_i}^2}\right) +
    tr\left(A^{-1}\frac{\partial^2{A}}{\partial{y_i}^2}\right) +
    tr\left(A^{-1}\frac{\partial^2{A}}{\partial{z_i}^2}\right) =
    tr(A^{-1} \Delta_{e_i} A)

to express the trace through sum using equality:

.. math::

    tr(AB) = \sum_{ij} a_{ij}b_{ji} = {a_i}^j {b_j}^i

notice that the :math:`\Delta_{e_i} A` has the only one non-zero :math:`row_i(\Delta_{e_i} A) = row_i(L)`:

.. math::

    tr(A^{-1} \Delta_{e_i} A) = {(A^{-1})_i}^j {(\Delta_{e_i} A)_j}^i

sum laplacian over :math:`i`:

.. math::

    \frac{\Delta \psi(\mathbf{r})}{\phi(\mathbf{r})} = (A^{-1})_{ij} L^{ji}

and get laplacian of multideterminant wavefunction:

.. math::

    \Delta \Phi(\mathbf{r}) / \Phi(\mathbf{r}) = \sum_n c_n \Delta \phi(\mathbf{r})_n / \sum_n c_n \phi(\mathbf{r})_n

where :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons

For certain electron coordinates, the laplacian can be obtained with casino.Slater.laplacian() method::

    slater.laplacian(n_vectors)

this is equivalent to (continues :ref:`from <inverse-matrix>`)::

    L_up, L_down = slater.laplacian_matrix(n_vectors)
    lap_up = np.einsum('ij,ji', inv_A_up, L_up)
    lap_down = np.einsum('ij,ji', inv_A_down, L_down)
    lap_up + lap_down

.. _hessian:

hessian
-------

Consider Slater determinant hessian by :math:`i`-th and :math:`j`-th electrons coordinates:

.. math::

    \frac{\nabla^2_{{e_i}{e_j}} \phi(\mathbf{r})}{\phi(\mathbf{r})} =
    tr(A^{-1} \nabla_{e_i} \nabla_{e_j} A - (A^{-1} \nabla_{e_i} A)(A^{-1} \nabla_{e_j} A))
    + \frac{\nabla_{e_i} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_j} \phi(\mathbf{r})}{\phi(\mathbf{r})}

to express the trace through sum using equality:

.. math::

    tr(AB) = \sum_{ij} a_{ij}b_{ji} = {a_i}^j {b_j}^i

notice that the :math:`\nabla_{e_i} A` has the only one non-zero :math:`row_i(\nabla_{e_i} A) = row_i(G)` and
the :math:`\nabla_{e_i} \nabla_{e_i} A` has only non-zero :math:`row_i(\nabla_{e_i} \nabla_{e_i} A) = row_i(H)`:

.. math::

    tr(A^{-1} \nabla_{e_i} \nabla_{e_j} A - (A^{-1} \nabla_{e_i} A)(A^{-1} \nabla_{e_j} A)) =
    {(A^{-1})_i}^j (\nabla_{e_i} {\nabla_{e_j} A)_j}^{iab} - {(A^{-1} \nabla_{e_i} A)_j}^{ia} {(A^{-1} \nabla_{e_j} A)_i}^{jb}

expand gradient vectors and hessian tensor over :math:`i` and :math:`j` (with Kronecker delta :math:`\delta_{ij}`):

.. math::

    \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} =
    \delta_{ij}{(A^{-1})_i}^j H_{jiab} - (A^{-1} G)_{jia} (A^{-1} G)_{ijb}
    + \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \\


we can get hessian of multideterminant wavefunction:

.. math::

    \nabla^2 \Phi(\mathbf{r}) / \Phi(\mathbf{r}) = \sum_n c_n \nabla^2 \phi(\mathbf{r})_n / \sum_n c_n \phi(\mathbf{r})_n

where :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons

For certain electron coordinates, the hessian matrix can be obtained with casino.Slater.hessian() method::

    slater.hessian(n_vectors)[0]

this is equivalent to (continues :ref:`from <inverse-matrix>`)::

    G_up, G_down = slater.gradient_matrix(n_vectors)
    tr_grad_u = np.einsum('ij,jia->ia', inv_A_up, G_up).reshape(neu * 3)
    tr_grad_d = np.einsum('ij,jib->ib', inv_A_down, G_down).reshape(ned * 3)
    mul_grad_u = np.einsum('ij,jka->ika', inv_A_up, G_up)
    mul_grad_d = np.einsum('ij,jkb->ikb', inv_A_down, G_down)
    grad = np.concatenate((tr_grad_u, tr_grad_d))

    H_up, H_down = slater.hessian_matrix(n_vectors)
    tr_hess_u = np.einsum('ij,jiab->iab', inv_A_up, H_up)
    tr_hess_d = np.einsum('ij,jiab->iab', inv_A_down, H_down)
    hess_u = np.einsum('ij,iab->iajb', np.eye(neu), tr_hess_u)
    hess_d = np.einsum('ij,iab->iajb', np.eye(ned), tr_hess_d)
    hess_u -= np.einsum('ijb,jia->iajb', mul_grad_u, mul_grad_u)
    hess_d -= np.einsum('ijb,jia->iajb', mul_grad_d, mul_grad_d)
    hess = np.zeros((ne * 3, ne * 3))
    hess[:neu * 3, :neu * 3] = hess_u.reshape(neu * 3, neu * 3)
    hess[neu * 3:, neu * 3:] = hess_d.reshape(ned * 3, ned * 3)
    hess += np.outer(grad, grad)

.. _tressian:

tressian
--------

Consider Slater determinant tressian by :math:`i`-th, :math:`j`-th and :math:`k`-th electrons coordinates:

.. math::

    \begin{align}
    & \frac{\nabla^3_{{e_i}{e_j}{e_k}} \phi(\mathbf{r})}{\phi(\mathbf{r})} = tr(A^{-1} \nabla_{e_i} \nabla_{e_j} \nabla_{e_k} A) - 2 \cdot \frac{\nabla_{e_i} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_j} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_k} \phi(\mathbf{r})}{\phi(\mathbf{r})} \\
    & + \frac{\nabla^2_{{e_i}{e_j}} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_k} \phi(\mathbf{r})}{\phi(\mathbf{r})} + \frac{\nabla^2_{{e_i}{e_k}} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_j} \phi(\mathbf{r})}{\phi(\mathbf{r})} + \frac{\nabla^2_{{e_j}{e_k}} \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla_{e_i} \phi(\mathbf{r})}{\phi(\mathbf{r})} \\
    & - tr((A^{-1} \nabla_{e_i} \nabla_{e_j} A)(A^{-1} \nabla_{e_k} A) + (A^{-1} \nabla_{e_i} \nabla_{e_k} A)(A^{-1} \nabla_{e_j} A) + (A^{-1} \nabla_{e_j} \nabla_{e_k} A)(A^{-1} \nabla_{e_i} A)) \\
    & + tr((A^{-1} \nabla_{e_i} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_k} A)) + tr((A^{-1} \nabla_{e_k} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_i} A))
    \end{align}

noting that:

.. math::

    tr((A^{-1} \nabla_{e_i} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_k} A)) = tr((A^{-1} \nabla_{e_k} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_i} A))

to express the trace through sum using equalities:

.. math::

    tr(AB) = \sum_{ij} a_{ij}b_{ji} = {a_i}^j {b_j}^i

.. math::

    tr(ABC) = \sum_{ijk} a_{ij}b_{jk}c_{ki} = {a_i}^j {b_j}^k {c_k}^i

.. math::

    \begin{align}
    & tr(A^{-1} \nabla_{e_i} \nabla_{e_j} \nabla_{e_k} A) \\
    & - tr((A^{-1} \nabla_{e_i} \nabla_{e_j} A)(A^{-1} \nabla_{e_k} A) + (A^{-1} \nabla_{e_i} \nabla_{e_k} A)(A^{-1} \nabla_{e_j} A) + (A^{-1} \nabla_{e_j} \nabla_{e_k} A)(A^{-1} \nabla_{e_i} A)) \\
    & + tr((A^{-1} \nabla_{e_i} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_k} A) + (A^{-1} \nabla_{e_k} A)(A^{-1} \nabla_{e_j} A)(A^{-1} \nabla_{e_i} A)) \\
    & = {(A^{-1})_i}^j {(\nabla_{e_i} \nabla_{e_j} \nabla_{e_k} A)_j}^{iabc} - {(A^{-1} \nabla_{e_i} \nabla_{e_j} A)_i}^{jab}{(A^{-1} \nabla_{e_k} A)_j}^{ic} \\
    & - {(A^{-1} \nabla_{e_i} \nabla_{e_k} A)_i}^{jac}{(A^{-1} \nabla_{e_j} A)_j}^{ib} - {(A^{-1} \nabla_{e_j} \nabla_{e_k} A)_i}^{jbc}{(A^{-1} \nabla_{e_i} A)_j}^{ia} \\
    & + {(A^{-1} \nabla_{e_i} A)_j}^{ia}{(A^{-1} \nabla_{e_j} A)_k}^{jb}{(A^{-1} \nabla_{e_k} A)_i}^{kc} + {(A^{-1} \nabla_{e_i} A)_k}^{ia}{(A^{-1} \nabla_{e_j} A)_i}^{jb}{(A^{-1} \nabla_{e_k} A)_j}^{kc}
    \end{align}

notice that the :math:`\nabla_i A` has only non-zero :math:`row_i(\nabla_i A) = row_i(G)` and
the :math:`\nabla_i \nabla_i A` has only non-zero :math:`row_i(\nabla_i \nabla_i A) = row_i(H)` and
the :math:`\nabla_i \nabla_i \nabla_i A` has only non-zero :math:`row_i(\nabla_i \nabla_i \nabla_i A) = row_i(T)`
and expand gradient vectors, hessian and tressian tensors over :math:`i`, :math:`j`, :math:`k`:

.. math::

    \begin{align}
    & \frac{\nabla^3 \phi(\mathbf{r})}{\phi(\mathbf{r})} = \delta_{ijk}{(A^{-1})_i}^jT_{jiabc} - 2 \cdot \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \\
    & + \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} + \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} + \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} \otimes \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} \\
    & - \delta_{ij}(A^{-1} H)_{ijab}(A^{-1} G)_{ijc} - \delta_{jk}(A^{-1} H)_{jkac}(A^{-1} G)_{jkb} - \delta_{ki}(A^{-1} G)_{kia}(A^{-1} H)_{kibc} \\
    & + (A^{-1} G)_{jia}(A^{-1} G)_{kjb}(A^{-1} G)_{ikc} + (A^{-1} G)_{kia}(A^{-1} G)_{ijb}(A^{-1} G)_{jkc}
    \end{align}


we can get tressian of multideterminant wavefunction:

.. math::

    \nabla^3 \Phi(\mathbf{r}) / \Phi(\mathbf{r}) = \sum_n c_n \nabla^3 \phi(\mathbf{r})_n / \sum_n c_n \phi(\mathbf{r})_n

where :math:`\mathbf{r}=\{r_{1}...r_{N}\}` are the coordinates of the N spin-up and spin-down electrons

For certain electron coordinates, the tressian metrix can be obtained with casino.Slater.tressian() method::

    slater.tressian(n_vectors)[0]

this is equivalent to (continues :ref:`from <inverse-matrix>`)::

    G_up, G_down = slater.gradient_matrix(n_vectors)
    tr_grad_u = np.einsum('ij,jia->ia', inv_A_up, G_up).reshape(neu * 3)
    tr_grad_d = np.einsum('ij,jib->ib', inv_A_down, G_down).reshape(ned * 3)
    grad = np.concatenate((tr_grad_u, tr_grad_d))

    H_up, H_down = slater.hessian_matrix(n_vectors)
    tr_hess_u = np.einsum('ij,jiab->iab', inv_A_up, H_up)
    tr_hess_d = np.einsum('ij,jiab->iab', inv_A_down, H_down)
    mul_grad_u = np.einsum('ik,kja->ija', inv_A_up, G_up)
    mul_grad_d = np.einsum('ik,kjb->ijb', inv_A_down, G_down)
    hess_u = np.einsum('ij,iab->iajb', np.eye(neu), tr_hess_u)
    hess_d = np.einsum('ij,iab->iajb', np.eye(ned), tr_hess_d)
    hess_u -= np.einsum('ijb,jia->iajb', mul_grad_u, mul_grad_u)
    hess_d -= np.einsum('ijb,jia->iajb', mul_grad_d, mul_grad_d)
    hess = np.zeros((ne * 3, ne * 3))
    hess[:neu * 3, :neu * 3] = hess_u.reshape(neu * 3, neu * 3)
    hess[neu * 3:, neu * 3:] = hess_d.reshape(ned * 3, ned * 3)
    hess += np.outer(grad, grad)

    T_up, T_down = slater.tressian_matrix(n_vectors)
    tr_tress_u = np.einsum('ij,jiabc->iabc', inv_A_up, T_up)
    tr_tress_d = np.einsum('ij,jiabc->iabc', inv_A_down, T_down)
    mul_hess_u = np.einsum('ik,kjab->iajb', inv_A_up, H_up)
    mul_hess_d = np.einsum('ik,kjab->iajb', inv_A_down, H_down)
    tress_u = np.einsum('ij,jk,iabc->iajbkc', np.eye(neu), np.eye(neu), tr_tress_u)
    tress_d = np.einsum('ij,jk,iabc->iajbkc', np.eye(ned), np.eye(ned), tr_tress_d)
    tress_u -= np.einsum('ij,kajb,jkc->iajbkc', np.eye(neu), mul_hess_u, mul_grad_u)
    tress_u -= np.einsum('ki,jaic,ijb->iajbkc', np.eye(neu), mul_hess_u, mul_grad_u)
    tress_u -= np.einsum('jk,ibkc,kia->iajbkc', np.eye(neu), mul_hess_u, mul_grad_u)
    tress_d -= np.einsum('ij,kajb,jkc->iajbkc', np.eye(ned), mul_hess_d, mul_grad_d)
    tress_d -= np.einsum('ki,jaic,ijb->iajbkc', np.eye(ned), mul_hess_d, mul_grad_d)
    tress_d -= np.einsum('jk,ibkc,kia->iajbkc', np.eye(ned), mul_hess_d, mul_grad_d)
    tress_u += 2 * np.einsum('jia,kjb,ikc->iajbkc', mul_grad_u, mul_grad_u, mul_grad_u)
    tress_d += 2 * np.einsum('jia,kjb,ikc->iajbkc', mul_grad_d, mul_grad_d, mul_grad_d)
    # tress_u += np.einsum('kia,ijb,jkc->iajbkc', mul_grad_u, mul_grad_u, mul_grad_u)
    # tress_d += np.einsum('kia,ijb,jkc->iajbkc', mul_grad_d, mul_grad_d, mul_grad_d)
    tress = np.zeros((ne * 3, ne * 3, ne * 3))
    tress[:neu * 3, :neu * 3, :neu * 3] = tress_u.reshape(neu * 3, neu * 3, neu * 3)
    tress[neu * 3:, neu * 3:, neu * 3:] = tress_d.reshape(ned * 3, ned * 3, ned * 3)
    tress += (
        np.einsum('i,jk->ijk', grad, hess) +
        np.einsum('k,ij->ijk', grad, hess) +
        np.einsum('j,ki->ijk', grad, hess) -
        2 * np.einsum('i,j,k->ijk', grad, grad, grad)
    )

Implementation
~~~~~~~~~~~~~~

The tressian tensor :math:`T[a, b, c]` has shape ``(N_e \cdot 3, N_e \cdot 3, N_e \cdot 3)``
where :math:`N_e` is the number of electrons of a given spin.  A naïve implementation
iterates over all triples :math:`(e_1, e_2, e_3)` with conditional checks inside the loop body:

.. code-block:: python

    for e1 in range(neu):
        for e2 in range(neu):
            for e3 in range(neu):
                res = 0
                if e1 == e2 == e3:
                    res += tr_tress[e1, ...]
                if e1 == e2:
                    res -= matrix_hess[e3, e1, ...] * matrix_grad[e1, e3, ...]
                # ... etc
                tress[e1*3+r1, e2*3+r2, e3*3+r3] += c * res

This costs :math:`O(N_e^6 \cdot 27)` iterations and the runtime branches prevent
LLVM from auto-vectorising the inner loops.

The key structural observation is that the determinant-specific contributions to
:math:`T[a, b, c]` are **sparse in the electron indices**: each term is non-zero
only when at least two of :math:`e_1, e_2, e_3` coincide.  Specifically:

- :math:`\mathrm{tr}(A^{-1} \nabla^3 A)` — non-zero only when :math:`e_1 = e_2 = e_3`
- :math:`\mathrm{tr}(A^{-1} \nabla^2_{e_i e_j} A \cdot A^{-1} \nabla_{e_k} A)` — non-zero only when :math:`e_i = e_j` (one constrained pair, :math:`e_k` free)
- :math:`(A^{-1}G)_{e_3 e_2}(A^{-1}G)_{e_1 e_3}(A^{-1}G)_{e_2 e_1}` — non-zero for **all** :math:`(e_1, e_2, e_3)`

This motivates decomposing the computation into five **branch-free** loop nests:

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Loop nest
     - Complexity
     - Replaces
   * - :math:`e_1 = e_2 = e_3 = e` (tr_tress diagonal)
     - :math:`O(N_e \cdot 27)`
     - ``if e1 == e2 == e3``
   * - :math:`e_1 = e_2`, free :math:`e_3` (hess×grad, pair 12)
     - :math:`O(N_e^2 \cdot 27)`
     - ``if e1 == e2``
   * - :math:`e_1 = e_3`, free :math:`e_2` (hess×grad, pair 13)
     - :math:`O(N_e^2 \cdot 27)`
     - ``if e1 == e3``
   * - :math:`e_2 = e_3`, free :math:`e_1` (hess×grad, pair 23)
     - :math:`O(N_e^2 \cdot 27)`
     - ``if e2 == e3``
   * - all :math:`(e_1, e_2, e_3)` free (triple product)
     - :math:`O(N_e^3 \cdot 27)`
     - always executed

Because no loop nest carries runtime conditionals, LLVM auto-vectorises all five
loops via SIMD instructions.  The outer-product term
:math:`\nabla\phi \otimes H + H \otimes \nabla\phi + \nabla\phi \otimes \nabla\phi \otimes \nabla\phi`
involves all :math:`(N_e \cdot 3)^3` elements and is unchanged.

.. code-block:: python

    # tr_tress: e1 == e2 == e3
    for e in range(neu):
        for r1, r2, r3 in product(range(3), repeat=3):
            tress[e*3+r1, e*3+r2, e*3+r3] += c * tr_tress_u[e, r1, r2, r3]

    # hess × grad: e1 == e2, free e3
    for e12 in range(neu):
        for e3 in range(neu):
            for r1, r2, r3 in product(range(3), repeat=3):
                tress[e12*3+r1, e12*3+r2, e3*3+r3] -= (
                    c * matrix_hess_u[e3, e12, r1, r2] * matrix_grad_u[e12, e3, r3]
                )

    # ... similarly for (e1==e3, free e2) and (e2==e3, free e1) ...

    # triple product: all (e1, e2, e3) free
    for e1 in range(neu):
        for e2 in range(neu):
            for e3 in range(neu):
                for r1, r2, r3 in product(range(3), repeat=3):
                    tress[e1*3+r1, e2*3+r2, e3*3+r3] += (
                        2 * c
                        * matrix_grad_u[e3, e2, r2]
                        * matrix_grad_u[e1, e3, r3]
                        * matrix_grad_u[e2, e1, r1]
                    )

On the He atom (:math:`N_e = 1` per spin, ``ne3 = 6``) the refactored implementation
is **5.5× faster** than the original (33 µs vs 184 µs per call).

The actual wall-clock speedup depends on system size.  The total cost of
:py:meth:`casino.Slater.tressian` is:

.. math::

    t_\text{total} = t_\text{matrix calls} + t_\text{index loops}

where :math:`t_\text{matrix calls}` is the combined time of
``value_matrix`` + ``gradient_matrix`` + ``hessian_matrix`` + ``tressian_matrix``
(all scale with system size and basis set), and :math:`t_\text{index loops}` is
the time of the electron-index loop nests described above.

For small systems (He, :math:`N_e = 1`) the index loops dominate and the 5.5×
speedup is realised.  For larger systems such as Ar (:math:`N_e = 9` per spin)
the matrix calls dominate — in particular ``tressian_matrix`` grows with the number
of orbitals and primitives — and the loop optimisation contributes only ~20% of
the total runtime.  The next bottleneck to address for large-:math:`N_e` systems
is therefore ``tressian_matrix`` itself.

