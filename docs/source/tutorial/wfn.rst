.. _wfn:

Wavefunction
============

Wavefunction in SJB Slater-Jastrow-backflow form is represented by the :class:`casino.Wfn` class.

.. math::

    \Psi(\mathbf{r}) = e^{J(\mathbf{r})}\Phi(\mathbf{X}(\mathbf{r}))

where :math:`\mathbf{X}(\mathbf{r}) = \mathbf{r} + \xi(\mathbf{r})` is collective coordinates, which depends on the configuration of the whole system.

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
   * - :ref:`value <value>`
     - :math:`\Psi(r)`
     - :math:`scalar`
   * - :ref:`coulomb <coulomb>`
     - :math:`V_{coul}`
     - :math:`scalar`
   * - :ref:`nonlocal_potential <nonlocal_potential>`
     - :math:`V_{nl}`
     - :math:`scalar`
   * - :ref:`kinetic_energy <kinetic_energy>`
     - :math:`T_{kin}`
     - :math:`scalar`
   * - :ref:`energy <energy>`
     - :math:`\hat H \Psi(r) / \Psi(r)`
     - :math:`scalar`
   * - :ref:`value_parameters_d1 <value_parameters_d1>`
     - :math:`\partial \ln \Psi(r) / \partial \alpha_i`
     - :math:`(N_{par} ,)`
   * - :ref:`kinetic_energy_parameters_d1 <kinetic_energy_parameters_d1>`
     - :math:`\partial \ln \Psi(r) / \partial \alpha_i`
     - :math:`(N_{par} ,)`
   * - :ref:`nonlocal_energy_parameters_d1 <nonlocal_energy_parameters_d1>`
     - :math:`\partial \ln \Psi(r) / \partial \alpha_i`
     - :math:`(N_{par} ,)`

.. _value:

value
-----

Value of wavefunction:

.. math::

    \Psi(\mathbf{r}) = e^{J(\mathbf{r})}\Phi(\mathbf{r} + \xi(\mathbf{r}))

.. _coulomb:

coulomb
-------

Value of e-e, e-n and n-n coulomb interaction:

.. math::

    V_{coul} =  \sum_i^{N_e} \sum_{j \gt i}^{N_e} \frac{1}{r_{ij}} - \sum_i^{N_e} \sum_\alpha^{N_I} \frac{Z_\alpha}{r_{i\alpha}} + \sum_\alpha^{N_I} \sum_{\beta \gt \alpha}^{N_I} \frac{Z_\alpha Z_\beta}{r_{\alpha\beta}}


.. _nonlocal_potential:

nonlocal_potential
------------------

Value of nonlocal potential:

.. math::

    V_{nl} = \sum_i^{N_e} \sum_l \frac{2l + 1}{4\pi} V_l(r_i) \int_{4\pi} P_l(cos \ \theta'_{i}) \frac{\Psi(\mathbf{r}\rvert_{r_i \to r_i'})}{\Psi(\mathbf{r})} d\Omega'_i

where :math:`P_l` denotes a Legendre polynomial, :math:`V_l(r_i)` is pseudopotential and :math:`\theta'_{i}=\angle(r_i, r_i')`.
Integral of the function :math:`f` defined on th unit sphere is approximated as:

.. math::

    \frac{1}{4\pi} \int_{4\pi} f(\Omega) d\Omega \approx \sum_q c_q f(r_q)

The values of the coefficients :math:`c_q` and unit vectors :math:`r_q` can be found in the appendix to the article
`Nonlocal pseudopotentials and diffusion Monte Carlo <https://people.physics.illinois.edu/Ceperley/papers/061.pdf>`_

after which we obtain:

.. math::

    V_{nl} = \sum_i^{N_e} \sum_l (2l + 1) V_l(r_i) \sum_q c_q P_l(cos \ \theta'_{i,q}) \frac{\Psi(\mathbf{r}\rvert_{r_i \to r_{i,q}'})}{\Psi(\mathbf{r})}

.. _kinetic_energy:

kinetic_energy
--------------

Value of kinetic energy usually represented as the sum of two terms:

.. math::

    T_{kin} = -\frac{1}{2} \sum_i^{N_e} \frac{\nabla_i^2 \Psi}{\Psi} = \sum_i^{N_e} 2T_i - F_i^2

where :math:`T_i` and :math:`F_i` are expressed through the logarithm of the modulus of the wave function:

.. math::

    T_i = - \frac{1}{4} \nabla_i^2 \ln \Psi = - \frac{1}{4} \left[ \frac{\nabla_i^2 \Psi}{\Psi} - \left(\frac{\nabla_i \Psi}{\Psi}\right)^2 \right]

.. math::

    F_i = \frac{1}{\sqrt{2}} \nabla_i \ln \Psi

This is convenient because the wave function is a product which leads to the following expressions:.

.. math::

    T_i = - \frac{1}{4} \left[ \frac{\nabla_i^2 \Phi}{\Phi} - \left(\frac{\nabla_i \Phi}{\Phi}\right)^2 + \nabla_i^2 J \right]

.. math::

    F_i = \frac{1}{\sqrt{2}} \left( \frac{\nabla_i \Phi}{\Phi} + \nabla_i J \right)

if backflow displacement :math:`\xi(\mathbf{r})` is not zero the coordinate transformation must be taken into account:

.. math::

    \nabla \Phi = \frac{\partial \Phi}{\partial \mathbf{X}} \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{r}}

.. math::

    \nabla^2 \Phi = tr\left(\left(\frac{\partial \mathbf{X}}{\partial \mathbf{r}}\right)^T
    \cdot \frac{\partial^2 \Phi}{\partial^2 \mathbf{X}}
    \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{r}}\right)
    + \frac{\partial \Phi}{\partial \mathbf{X}} \cdot \frac{\partial^2 \mathbf{X}}{\partial^2 \mathbf{r}}

.. _energy:

energy
------

Value of energy.

.. math::

    E = T_{kin} + V_{coul} + V_{nl}


.. _value_parameters_d1:

value_parameters_d1
-------------------

Partial derivatives of the wave function with respect to Jastrow, backflow and Slater the parameters :math:`\{\alpha_i^J, \alpha_i^B, \alpha_i^S \}`:

.. math::

    \frac{\partial \ln \Psi}{\partial \alpha} = \left(\frac{\partial J}{\partial \alpha_i^J} , \frac{\nabla_{\mathbf{X}} \Phi}{\Phi} \cdot \frac{\partial \mathbf{X}}{\partial \alpha_i^B} , \frac{\partial \ln \Phi}{\partial \alpha_i^S} \right)

.. _kinetic_energy_parameters_d1:

kinetic_energy_parameters_d1
----------------------------

Partial derivatives of the kinetic energy with respect to Jastrow, backflow and Slater the parameters :math:`\{\alpha_i^J, \alpha_i^B, \alpha_i^S \}`:

.. math::

    \frac{1}{2} \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha} = \frac{1}{2} \left(
        \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^J},
        \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^B},
        \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^S},
    \right)

.. math::

    \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^J} = \frac{\partial \nabla^2 J}{\partial \alpha_i^J} + 2 \frac{\nabla \Psi}{\Psi} \cdot \frac{\partial \nabla J}{\partial \alpha_i^J},

.. math::

    \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^B} =

.. math::

    \frac{\partial \nabla_i^2 \Psi / \Psi}{\partial \alpha^S} = \frac{\partial \nabla^2 \ln \Phi}{\partial \alpha^S} +
    2 \frac{\partial \nabla \ln \Phi}{\partial \alpha^S} \left[ \nabla J + \frac{\nabla \Phi}{\Phi} \right]

if backflow displacement :math:`\xi(\mathbf{r})` is not zero the coordinate transformation must be taken into account:

.. math::

    \nabla \Phi = \frac{\partial \Phi}{\partial \mathbf{X}} \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{r}}

.. math::

    \nabla^2 \Phi = tr\left(\left(\frac{\partial \mathbf{X}}{\partial \mathbf{r}}\right)^T
    \cdot \frac{\partial^2 \Phi}{\partial^2 \mathbf{X}}
    \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{r}}\right)
    + \frac{\partial \Phi}{\partial \mathbf{X}} \cdot \frac{\partial^2 \mathbf{X}}{\partial^2 \mathbf{r}}

.. _nonlocal_energy_parameters_d1:

nonlocal_energy_parameters_d1
-----------------------------

Partial derivatives of the nonlocal energy with respect to Jastrow, backflow and Slater the parameters:

.. math::

    \frac{\partial V_{nl}}{\partial \alpha} = \sum_i^{N_e} \sum_l (2l + 1) V_l(r_i) \sum_q c_q P_l \left( cos \ \theta'_{i,q} \right) \frac{\partial}{\partial \alpha}
    \left[ \frac{\Psi(\mathbf{r}\rvert_{r_i \to r_{i,q}'})}{\Psi(\mathbf{r})} \right]

.. math::

    = \sum_i^{N_e} \sum_l (2l + 1) V_l(r_i) \sum_q c_q P_l \left( cos \ \theta'_{i,q} \right) \frac{\Psi(\mathbf{r}\rvert_{r_i \to r_i'})}{\Psi(\mathbf{r})}
    \left[ \frac{\partial \ln \Psi(\mathbf{r}\rvert_{r_i \to r_{i,q}'})}{\partial \alpha} - \frac{\partial \ln \Psi(\mathbf{r})}{\partial \alpha} \right]
