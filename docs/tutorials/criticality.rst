.. _tutorial-criticality:

Reactor criticality with the generic solver
===========================================

dualmesh has no neutronics module.  This tutorial writes the multigroup
neutron diffusion equations with ``coefficient_form_PDE`` and finds the
effective multiplication factor with the eigenvalue study of any problem.
The same steps apply to any eigenvalue problem that no physics names.  The
physics follows J. C. Lee, *Nuclear Reactor Physics and Engineering*
[Lee2025]_.

The equations
-------------

Diffusion theory divides the neutron energy range into :math:`G` groups.
Group 1 has the highest energy.  The scalar flux :math:`\phi_g` of group
:math:`g` satisfies the steady multigroup diffusion equation ([Lee2025]_,
Eq. 7.19):

.. math::
   :label: multigroup-diffusion

   -\nabla \cdot D_g \nabla \phi_g + \Sigma_{R,g}\, \phi_g
   - \sum_{h \neq g} \Sigma_{s,h \to g}\, \phi_h
   = \frac{\chi_g}{k} \sum_{h=1}^{G} \nu\Sigma_{f,h}\, \phi_h ,
   \qquad g = 1, \dots, G ,

where :math:`D_g` is the diffusion coefficient, :math:`\Sigma_{s,h \to g}`
the macroscopic cross section of scattering from group :math:`h` into group
:math:`g`, :math:`\nu\Sigma_{f,h}` the number of neutrons from one fission
multiplied by the fission cross section, :math:`\chi_g` the fraction of the
fission neutrons that start in group :math:`g` and :math:`k` the effective
multiplication factor.  The removal cross section is the absorption cross
section :math:`\Sigma_{a,g}` plus the scattering out of the group
([Lee2025]_, Eq. 7.18), :math:`\Sigma_{R,g} = \Sigma_{a,g} + \sum_{h \neq g}
\Sigma_{s,g \to h}`.

The equations have a solution that is not 0 only for specified values of
:math:`k`.  The largest value is the effective multiplication factor
:math:`k_\mathrm{eff}`, and its eigenvector, the only one that is positive
at all points, is the fundamental mode of the flux.

A two-dimensional model of a core of finite height leaves out the axial
direction.  The term :math:`D_g B_z^2 \phi_g` gives the axial leakage, where
:math:`B_z^2` is the axial buckling ([Lee2025]_, Eqs. 6.66 and 6.67, and
Sect. 6.7.2).  It adds :math:`D_g B_z^2` to the removal cross section.

Write the equations
-------------------

Each group is one field of ``coefficient_form_PDE``.  The removal is the
absorption coefficient of the equation of the group, and the scattering from
another group is an absorption coefficient that couples to that group, with
the opposite sign.  The fission source carries the symbol ``eigenvalue``,
which is :math:`1/k`.  The eigenvalue study then finds the smallest
eigenvalue, which gives the largest :math:`k`.  For two groups, with all
fission neutrons born fast:

.. code-block:: python

   diffusion = {"phi1": "D1", "phi2": "D2"}
   absorption = {
       "phi1": "Sigma_a1 + Sigma_12",
       "phi2": {"phi2": "Sigma_a2", "phi1": "-Sigma_12"},
   }
   fission = {"phi1": "eigenvalue*nu_Sigma_f2*phi2"}
   neutrons = problem.add_physics(
       "coefficient_form_PDE",
       "neutrons",
       variables=["phi1", "phi2"],
       diffusion_coefficient=diffusion,
       absorption_coefficient=absorption,
       source=fission,
       constants=cross_sections,
   )
   result = problem.solve_eigenvalue(num_modes=1)
   k = 1.0 / result.eigenvalues[0]

Give the cross sections in SI units, :math:`\mathrm{m}^{-1}`, and the
diffusion coefficients in :math:`\mathrm{m}`.  The script
``examples/reactor_criticality.py`` computes a bare square core with zero
flux on its boundary.  For such a core, two-group theory gives
([Lee2025]_, Eq. 7.26)

.. math::

   k = \frac{\nu\Sigma_{f,2}\, \Sigma_{1 \to 2}}{(\Sigma_{a,1} + \Sigma_{1 \to 2} + D_1 B^2)(\Sigma_{a,2} + D_2 B^2)} ,

where :math:`B^2 = 2 (\pi / L)^2` is the geometrical buckling of a square of
side :math:`L`.  The example gives this value to 1.5 pcm on a mesh of
:math:`40 \times 40` elements.

Regions with their own cross sections
-------------------------------------

A core has regions of different materials.  Give the cross sections of each
region as properties of a ``constant_property`` object on its block, and name
the properties in ``properties``.  Without a diffusion coefficient, the
equation of a field ``phi1`` reads the property
``diffusion_coefficient_phi1``:

.. code-block:: python

   problem.add_property(
       "constant_property",
       "fuel",
       block=["fuel"],
       property_names=["diffusion_coefficient_phi1", "diffusion_coefficient_phi2", "removal_1"],
       property_values=[0.015, 0.004, 3.0],
   )

The script ``verification/benchmarks/neutronics/run_iaea_2d_pwr.py`` writes
the IAEA PWR benchmark in this way.  :doc:`/verification` gives its results.

Boundary conditions
-------------------

At a free surface, no neutron comes back into the core, so the partial
current into the core is 0.  In diffusion theory this gives ([Lee2025]_,
Eq. 5.2)

.. math::

   -D_g \frac{\partial \phi_g}{\partial n} = \frac{\phi_g}{r} ,

with :math:`r = 2`, where :math:`\mathbf{n}` is the outward normal.  Transport
theory gives :math:`r = 2.1312` ([Lee2025]_, Sect. 5.1).  This condition is
a ``Robin_boundary_condition`` with ``transfer_coefficient`` equal to
:math:`1/r`.  An albedo :math:`\alpha`, the fraction of the outgoing partial
current that comes back, gives the transfer coefficient :math:`(1 -
\alpha)/(2(1 + \alpha))`.  A boundary without a condition has no net
current, which is the condition at a plane of symmetry.

Verification
------------

The tests in ``tests/python/test_criticality.py`` compare the eigenvalue
with the analytical solutions of one-group and two-group diffusion theory.
The flux of a bare homogeneous core is 0 at the boundary, and
:math:`k = \nu\Sigma_f / (\Sigma_a + D B_g^2)` ([Lee2025]_, Eq. 5.71), with
the geometrical buckling :math:`(\pi/H)^2` for a slab of thickness
:math:`H`, :math:`(2.405/R)^2` for an infinite cylinder of radius :math:`R`,
:math:`(\pi/R)^2` for a sphere and :math:`(2.405/R)^2 + (\pi/H)^2` for a
finite cylinder ([Lee2025]_, Sect. 5.3).  All four methods converge to the
slab value at second order in the element size.  A slab with the vacuum
condition has the flux :math:`\cos(Bx)` with :math:`D B \tan(B a) = 1/r`,
where :math:`a` is the half thickness, and the calculation gives it to
:math:`10^{-6}`.  For an infinite medium of the IAEA fuel,
:math:`k_\infty = 1.125` ([Lee2025]_, Eq. 7.27) to round-off.  The IAEA PWR
benchmark gives :math:`k_\mathrm{eff}` to 5.5 pcm with 8 finite elements per
assembly.
