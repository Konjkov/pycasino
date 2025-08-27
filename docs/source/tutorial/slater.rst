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

.. list-table::
   :widths: 30 30 40
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - :ref:`value_matrix <value-matrix>`
     - :math:`A^\uparrow, A^\downarrow`
     - :math:`(N^\uparrow, M), (N^\downarrow, M)`
   * - :ref:`gradient_matrix <gradient-matrix>`
     - :math:`G^\uparrow, G^\downarrow`
     - :math:`(N^\uparrow, M, 3), (N^\downarrow, M, 3)`
   * - :ref:`laplacian_matrix <laplacian-matrix>`
     - :math:`L^\uparrow, L^\downarrow`
     - :math:`(N^\uparrow, M), (N^\downarrow, M)`
   * - :ref:`hessian_matrix <hessian-matrix>`
     - :math:`H^\uparrow, H^\downarrow`
     - :math:`(N^\uparrow, M, 3, 3), (N^\downarrow, M, 3, 3)`
   * - :ref:`tressian_matrix <tressian-matrix>`
     - :math:`T^\uparrow, T^\downarrow`
     - :math:`(N^\uparrow, M, 3, 3, 3), (N^\downarrow, M, 3, 3, 3)`
   * - :ref:`value <value>`
     - :math:`\Psi(r)`
     - :math:`scalar`
   * - :ref:`gradient <gradient>`
     - :math:`\nabla \Psi(r)/\Psi(r)`
     - :math:`(3N,)`
   * - :ref:`laplacian <laplacian>`
     - :math:`\Delta \Psi(r)/\Psi(r)`
     - :math:`scalar`
   * - :ref:`hessian <hessian>`
     - :math:`\nabla^2 \Psi(r)/\Psi(r)`
     - :math:`(3N, 3N)`
   * - :ref:`tressian <tressian>`
     - :math:`\nabla^3 \Psi(r)/\Psi(r)`
     - :math:`(3N, 3N, 3N)`

Slater class has a following methods:

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
