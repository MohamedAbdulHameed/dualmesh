.. _theory-neutronics:

Neutron diffusion
=================

The neutrons in a reactor core set the power distribution of the core.  The
balance of the neutrons also sets if the chain reaction continues.  Transport
theory gives an accurate description of the neutrons.  But a reactor core is
large in relation to the mean free path of a neutron.  Thus, diffusion theory
is accurate in most of the core.

In diffusion theory, the neutron current is a linear function of the gradient
of the scalar flux.  Diffusion theory with two or more energy groups is the
usual method for the core calculations of light water reactors.  A transport
calculation of each fuel assembly supplies the group constants.

This chapter gives the equations of the ``neutron_diffusion`` physics, its
boundary conditions and the eigenvalue study.  The chapter follows J. C. Lee,
*Nuclear Reactor Physics and Engineering* [Lee2025]_.

Multigroup diffusion equations
------------------------------

The physics divides the neutron energy range into :math:`G` groups.  Group 1
has the highest energy.  The scalar flux :math:`\phi_g` of group :math:`g`
satisfies the steady multigroup diffusion equation ([Lee2025]_, Eq. 7.19):

.. math::
   :label: multigroup-diffusion

   -\nabla \cdot D_g \nabla \phi_g + \Sigma_{R,g}\, \phi_g
   - \sum_{h \neq g} \Sigma_{s,h \to g}\, \phi_h
   = \frac{\chi_g}{k} \sum_{h=1}^{G} \nu\Sigma_{f,h}\, \phi_h ,
   \qquad g = 1, \dots, G .

In this equation:

* :math:`D_g` is the diffusion coefficient.
* :math:`\Sigma_{s,h \to g}` is the macroscopic cross section of scattering
  from group :math:`h` into group :math:`g`.
* :math:`\nu\Sigma_{f,h}` is the number of neutrons from one fission,
  multiplied by the fission cross section.
* :math:`\chi_g` is the fraction of the fission neutrons that start in group
  :math:`g`.
* :math:`k` is the effective multiplication factor.

The removal cross section is the absorption cross section
:math:`\Sigma_{a,g}` plus the scattering out of the group ([Lee2025]_,
Eq. 7.18):

.. math::
   :label: removal-cross-section

   \Sigma_{R,g} = \Sigma_{a,g} + \sum_{h \neq g} \Sigma_{s,g \to h} .

Scattering in one group does not add or remove neutrons of that group.  Thus,
this scattering is not in the equations.  As matrices, the equations are
:math:`\mathbf{L} \boldsymbol{\phi} = k^{-1} \mathbf{F} \boldsymbol{\phi}`
([Lee2025]_, Eq. 7.20).  The matrix :math:`\mathbf{L}` contains the leakage,
the removal and the scattering between groups.  The matrix :math:`\mathbf{F}`
contains the fission source.

The physics makes one variable for each group, from ``neutron_flux_1`` to
``neutron_flux_G``.  For each group, the physics adds these kernels:

* A ``diffusion`` kernel for the leakage.
* A ``reaction`` kernel for the removal.
* A ``coupled_force`` kernel for the scattering from each other group.
* A ``coupled_force`` kernel for the fission source of each group.

The kernels read the cross sections from the ``multigroup_cross_sections``
property object of each region.  This property object declares
:math:`D_g`, :math:`\Sigma_{R,g}`, :math:`\Sigma_{s,h \to g}`,
:math:`\nu\Sigma_{f,g}` and the fission production
:math:`\chi_g \nu\Sigma_{f,h}`:

.. code-block:: python

   neutrons = problem.add_physics("neutron_diffusion", "neutrons", groups=2)
   problem.add_property(
       "multigroup_cross_sections",
       "fuel",
       block=["fuel"],
       diffusion_coefficient=[0.015, 0.004],
       absorption_cross_section=[1.0, 8.0],
       scattering_cross_section=[[0.0, 2.0], [0.0, 0.0]],
       nu_fission_cross_section=[0.0, 13.5],
   )
   neutrons.add_boundary_condition("vacuum_boundary_condition", "outer")
   result = problem.solve_eigenvalue()

Give the cross sections in SI units, :math:`\mathrm{m}^{-1}`, and the
diffusion coefficients in :math:`\mathrm{m}`.  The scattering matrix has one
row for each source group.

**Transverse buckling.**  A two-dimensional model of a core of finite height
does not include the axial direction.  The term :math:`D_g B_z^2 \phi_g`
gives the axial leakage, where :math:`B_z^2` is the axial buckling.  This
term replaces the absorption cross section with :math:`\Sigma_a + D B_z^2`
([Lee2025]_, Eqs. 6.66 and 6.67, and Sect. 6.7.2).  The parameter
``transverse_buckling`` adds this term to all groups.

Boundary conditions
-------------------

At a free surface, no neutron comes back into the core.  Thus, the partial
current into the core is 0.  In diffusion theory, the partial current into the
surface is :math:`J^- = \phi / 4 + (D/2)\, \partial \phi / \partial n`, where
:math:`\mathbf{n}` is the outward normal.  The condition :math:`J^- = 0`
gives ([Lee2025]_, Eq. 5.2):

.. math::
   :label: vacuum-condition

   -D_g \frac{\partial \phi_g}{\partial n} = \frac{\phi_g}{r} ,

with :math:`r = 2`.

A linear extrapolation of the flux with this slope gives 0 at the distance
:math:`r D_g` from the surface ([Lee2025]_, Eqs. 5.3 and 5.4).  Transport
theory puts this point at :math:`0.7104\, \lambda_{tr} = 2.1312\, D_g`, where
:math:`\lambda_{tr} = 3D` is the transport mean free path ([Lee2025]_,
Sect. 5.1).

The physics has these boundary conditions:

* ``vacuum_boundary_condition`` applies Eq. :eq:`vacuum-condition` with the
  ratio ``extrapolation_distance_ratio``.  The default ratio is 2.  The ratio
  2.1312 gives the condition :math:`\partial \phi_g / \partial n = -0.4692\,
  \phi_g / D_g` of the IAEA benchmarks.
* ``albedo_boundary_condition`` sends back the fraction :math:`\alpha` of the
  partial current out of the core.  This condition gives
  :math:`-D_g\, \partial \phi_g / \partial n = \phi_g (1 - \alpha)/(2(1 +
  \alpha))`.
* ``Dirichlet_boundary_condition`` with ``value=0.0`` sets all fluxes to 0.

A boundary without a condition has no net current.  This is the condition at
a plane of symmetry.

The eigenvalue study
--------------------

The equations :eq:`multigroup-diffusion` have a solution that is not 0 only
for specified values of :math:`k`.  The largest value is the effective
multiplication factor :math:`k_\mathrm{eff}`.  Its eigenvector is the
fundamental mode of the flux.  This eigenvector is the only eigenvector that
is positive at all points.

:meth:`~dualmesh.Problem.solve_eigenvalue` assembles the two discrete
operators from the objects of the problem.  The Jacobian without the fission
source is :math:`\mathbf{L}`.  The Jacobian with the fission source is
:math:`\mathbf{L} - \mathbf{F}`.  The study factorizes :math:`\mathbf{L}` one
time.  Then it computes the dominant eigenvalue of
:math:`\mathbf{L}^{-1}\mathbf{F}` with one of two methods:

* ``method="power"`` is the source iteration of [Lee2025]_, Eqs. 6.40 and
  6.44.  Each iteration solves :math:`\mathbf{L} \boldsymbol{\phi}^{(i+1)} =
  \mathbf{F} \boldsymbol{\phi}^{(i)} / k^{(i)}`.  The new value
  :math:`k^{(i+1)}` is :math:`k^{(i)}` multiplied by the ratio of the total
  fission sources of the two iterates.

  In each iteration, the error decreases by the dominance ratio
  :math:`k_1 / k_0`, where :math:`k_1` is the second eigenvalue.  A large core
  has a dominance ratio near 1, and then the method needs many iterations.
* ``method="krylov"`` is the default.  It applies the Arnoldi method with
  implicit restarts (ARPACK through SciPy, [Lee2025]_, Sect. 6.9) to
  :math:`\mathbf{L}^{-1}\mathbf{F}`.  When the dominance ratio is near 1, this
  method needs far fewer applications of the operator.

The study then scales the flux.  After this step, the total production of
fission neutrons, :math:`\int \sum_g \nu\Sigma_{f,g} \phi_g \, dV`, is equal
to ``normalization``.  The default value is 1.  The two methods agree to the
tolerance of the eigenvalue.  The tests in ``tests/python/test_neutronics.py``
make sure of this agreement.

Verification
------------

The tests in ``tests/python/test_neutronics.py`` compare the eigenvalue with
the analytical solutions of one-group and two-group diffusion theory.

**Bare cores.**  The flux of a bare homogeneous core is 0 at the boundary.
Its fundamental mode satisfies the Helmholtz equation
:math:`\nabla^2 \phi + B_g^2 \phi = 0`, and ([Lee2025]_, Eq. 5.71):

.. math::

   k = \frac{\nu\Sigma_f}{\Sigma_a + D B_g^2} .

The geometrical buckling :math:`B_g^2` has these values ([Lee2025]_,
Sect. 5.3):

* :math:`(\pi/H)^2` for a slab of thickness :math:`H`.
* :math:`(2.405/R)^2` for an infinite cylinder of radius :math:`R`.
* :math:`(\pi/R)^2` for a sphere of radius :math:`R`.
* :math:`(2.405/R)^2 + (\pi/H)^2` for a finite cylinder of height
  :math:`H`.

The test values are :math:`D = 1\ \mathrm{cm}`, :math:`\Sigma_a = 0.02\
\mathrm{cm}^{-1}` and :math:`\nu\Sigma_f = 0.03\ \mathrm{cm}^{-1}`.  For a slab
of 1 m, :math:`k = 1.4294590`.  All methods converge to this value with the
second order in the element size.

**Vacuum condition.**  A slab with the condition :eq:`vacuum-condition` has
the flux :math:`\cos(Bx)` with :math:`D B \tan(B a) = 1/r`, where :math:`a` is
the half thickness.  The calculation gives this solution to
:math:`10^{-6}` for :math:`r = 2` and for :math:`r = 2.1312`.

**Two groups.**  The tests use the fuel of the IAEA benchmark.  For an
infinite medium, :math:`k_\infty = \nu\Sigma_{f,2} \Sigma_{1 \to 2} /
((\Sigma_{a,1} + \Sigma_{1 \to 2}) \Sigma_{a,2}) = 1.125` ([Lee2025]_,
Eq. 7.27).  The ratio of the thermal flux to the fast flux is
:math:`\Sigma_{1 \to 2} / \Sigma_{a,2} = 0.25`.  The calculation gives these
values to the round-off error.  For a bare square core with a transverse
buckling ([Lee2025]_, Eq. 7.26, with the same buckling in the two groups):

.. math::

   k = \frac{\nu\Sigma_{f,2}\, \Sigma_{1 \to 2}}{(\Sigma_{a,1} + \Sigma_{1 \to 2} + D_1 B^2)(\Sigma_{a,2} + D_2 B^2)} .

The calculation gives this value to :math:`10^{-4}`.

The benchmarks in :doc:`/verification` compare the physics with the reference
solutions of the IAEA reactor benchmarks.
