.. _tutorial-criticality:

Reactor criticality with the generic solver
===========================================

dualmesh-multiphysics has no neutronics module.  In this tutorial, you write
the multigroup neutron diffusion equations as text with the
``general_form_PDE`` physics, and you find the effective multiplication
factor of a reactor core with the eigenvalue study that every problem has.
The same steps solve any eigenvalue problem that no physics names.  The
physics follows J. C. Lee, *Nuclear Reactor Physics and Engineering*
[Lee2025]_.

The equations
-------------

Diffusion theory divides the energy range of the neutrons into :math:`G`
groups, where group 1 has the highest energy.  The scalar flux
:math:`\phi_g` of group :math:`g`, in :math:`\mathrm{1/(m^2\,s)}`, satisfies
the steady multigroup diffusion equation ([Lee2025]_, Eq. 7.19)

.. math::
   :label: multigroup-diffusion

   -\nabla \cdot D_g \nabla \phi_g + \Sigma_{R,g}\, \phi_g
   - \sum_{h \neq g} \Sigma_{s,h \to g}\, \phi_h
   = \frac{\chi_g}{k} \sum_{h=1}^{G} \nu\Sigma_{f,h}\, \phi_h ,
   \qquad g = 1, \dots, G .

The symbols are:

* :math:`D_g`, the diffusion coefficient of group :math:`g`, in
  :math:`\mathrm{m}`.
* :math:`\Sigma_{s,h \to g}`, the macroscopic cross section of scattering
  from group :math:`h` into group :math:`g`, in :math:`\mathrm{1/m}`.
* :math:`\nu\Sigma_{f,h}`, one symbol for the number of neutrons that one
  fission releases times the fission cross section of group :math:`h`, in
  :math:`\mathrm{1/m}`.
* :math:`\chi_g`, the fraction of the fission neutrons that start in group
  :math:`g`.
* :math:`k`, the multiplication factor.
* :math:`\Sigma_{R,g} = \Sigma_{a,g} + \sum_{h \neq g} \Sigma_{s,g \to h}`,
  the removal cross section, which adds the scattering out of the group to
  the absorption cross section :math:`\Sigma_{a,g}` ([Lee2025]_, Eq. 7.18).

The equations have a solution other than zero only for some values of
:math:`k`.  The largest is the effective multiplication factor
:math:`k_\mathrm{eff}`, and its eigenvector, the only one that is positive
at every point, is the fundamental mode of the flux.

A two-dimensional model of a core of finite height leaves out the axial
direction.  The term :math:`D_g B_z^2 \phi_g` gives the axial leakage, where
:math:`B_z^2` is the axial buckling in :math:`\mathrm{1/m^2}` ([Lee2025]_,
Eqs. 6.66 and 6.67, and Sect. 6.7.1).  It adds :math:`D_g B_z^2` to the
removal cross section.

Write the equations
-------------------

The script ``examples/reactor_criticality.py`` computes a bare square core of
side :math:`L = 2\ \mathrm{m}` with zero flux on its boundary, in two groups,
with all fission neutrons born fast.

1. Create the problem on a mesh of :math:`40 \times 40` elements:

   .. literalinclude:: ../../examples/reactor_criticality.py
      :language: python
      :start-at: mesh = dualmesh.generate_rectangle_mesh(
      :end-at: problem = dualmesh.Problem(mesh, method=method)
      :dedent: 4

   The function ``criticality`` of the script takes ``num_elements=40`` and
   ``method="FEM"`` by default.

2. Give the cross sections in SI units, the cross sections in
   :math:`\mathrm{1/m}` and the diffusion coefficients in
   :math:`\mathrm{m}`:

   .. literalinclude:: ../../examples/reactor_criticality.py
      :language: python
      :start-at: cross_sections = {
      :end-at: }

3. Write one equation per group as it is printed.  The fission source
   carries the symbol ``eigenvalue``, which is :math:`1/k`, so the smallest
   eigenvalue gives the largest :math:`k`:

   .. literalinclude:: ../../examples/reactor_criticality.py
      :language: python
      :start-at: neutrons = problem.add_physics(
      :end-before: boundary = list(mesh.sideset_names())
      :dedent: 4

   Every keyword of ``add_physics`` that is not a parameter, here each cross
   section, is a named constant that the equations use.  Each equation
   belongs to the field whose divergence it holds.  ``units`` gives the unit
   of each flux, which the output files carry.

4. Hold the flux at zero on the boundary, and solve the eigenvalue problem
   for one mode:

   .. literalinclude:: ../../examples/reactor_criticality.py
      :language: python
      :start-at: boundary = list(mesh.sideset_names())
      :end-at: return 1.0 / float(result.eigenvalues[0])
      :dedent: 4

For a bare core, two-group theory without fast fission gives
([Lee2025]_, Eq. 7.26, without up-scattering and with one buckling for both
groups)

.. math::

   k = \frac{\nu\Sigma_{f,2}\, \Sigma_{s,1 \to 2}}
   {(\Sigma_{a,1} + \Sigma_{s,1 \to 2} + D_1 B^2)(\Sigma_{a,2} + D_2 B^2)} ,

where :math:`B^2 = 2 (\pi / L)^2` is the geometrical buckling of the square,
in :math:`\mathrm{1/m^2}`.  The function ``exact_criticality`` of the script
computes it, and the script prints the computed :math:`k`, the exact one and
their difference in pcm, where one pcm (per cent mille) is :math:`10^{-5}`
in :math:`k`.  The test ``test_the_example_matches_two_group_theory`` in
``tests/python/test_criticality.py`` runs the script on 10, 20 and 40
elements per side.  It checks that the order of convergence of the error exceeds
1.8 and that the error is below :math:`2 \times 10^{-5}\,k` on the finest
mesh.

Regions with their own cross sections
-------------------------------------

A core has regions of different materials.  Give the cross sections of each
region as properties of a ``constant`` property object on its block.  The
equations use them by their names, which take the values of the region at
every point.  The script ``verification/benchmarks/neutronics/run_iaea_2d_pwr.py``
writes the two-dimensional benchmark of the International Atomic Energy
Agency (IAEA) for a pressurized water reactor (PWR) [ANL7416]_ in this way.
It adds one property object per region, with the axial buckling in the
removal cross sections.  In its loop over the regions, ``d1``, ``d2``,
``s12``, ``a1``, ``a2`` and ``nf2`` are the constants of the region in SI
units, and ``buckling`` is the axial buckling:

.. literalinclude:: ../../verification/benchmarks/neutronics/run_iaea_2d_pwr.py
   :language: python
   :start-at: problem.add_property(
   :end-before: # The cross sections are the properties of the blocks
   :dedent: 8

and writes the equations with these names:

.. literalinclude:: ../../verification/benchmarks/neutronics/run_iaea_2d_pwr.py
   :language: python
   :start-at: neutrons = problem.add_physics(
   :end-before: # No incoming current
   :dedent: 4

:doc:`/verification` gives the results of the benchmark.

Boundary conditions
-------------------

At a free surface, no neutron comes back into the core: the partial current
into the core is zero.  In diffusion theory this gives ([Lee2025]_, Eq. 5.2,
written with :math:`D_g` in place of a third of the transport mean free
path)

.. math::

   -D_g \nabla\phi_g \cdot \mathbf{n} = \frac{\phi_g}{2} ,

where :math:`\mathbf{n}` is the outward unit normal.  Transport theory places
the extrapolated boundary at :math:`0.7104` transport mean free paths outside
the surface ([Lee2025]_, Sect. 5.1), which replaces the 2 by :math:`3 \times
0.7104 = 2.1312`.  Write the condition as it is printed, with the flux of the
equation of the group on the left:

.. literalinclude:: ../../verification/benchmarks/neutronics/run_iaea_2d_pwr.py
   :language: python
   :start-at: neutrons.add_boundary_condition("-diffusion_1
   :end-at: neutrons.add_boundary_condition("-diffusion_2
   :dedent: 4

``dphi1/dn`` is the derivative of ``phi1`` along the outward normal.  A
boundary where a fraction :math:`a_b` of the outgoing partial current comes
back, the albedo, takes :math:`-D_g \nabla\phi_g \cdot \mathbf{n} = \phi_g
(1 - a_b)/(2(1 + a_b))`.  A boundary without a condition has no net current,
which is the condition at a plane of symmetry.

Check the method
----------------

The tests in ``tests/python/test_criticality.py`` compare the eigenvalue
with the solutions of one-group and two-group diffusion theory.  A bare
homogeneous core has zero flux at its boundary, and one-group theory, with the
coefficients :math:`D`, :math:`\Sigma_a` and :math:`\nu\Sigma_f` of the one
group, gives
:math:`k = \nu\Sigma_f / (\Sigma_a + D B_g^2)` ([Lee2025]_, Eq. 5.71), where
:math:`B_g^2` is the geometrical buckling ([Lee2025]_, Sect. 5.3.3 and Table
5.2): :math:`(\pi/H)^2` for a slab of thickness :math:`H`,
:math:`(\pi/R)^2` for a sphere of radius :math:`R`, and :math:`(2.405/R)^2 +
(\pi/H)^2` for a cylinder of radius :math:`R` and height :math:`H`, which
gives :math:`(2.405/R)^2` for an infinite cylinder.  The tests check that:

* On a slab, the order of convergence of :math:`k` in the element size
  exceeds 1.9 with both the finite element method (FEM) and the dual mesh
  control domain method (DMCDM), and the flux is the cosine of the
  fundamental mode.
* On an infinite cylinder, a sphere and a finite cylinder, :math:`k` agrees
  with the buckling formula to a relative :math:`3 \times 10^{-5}`.
* On a slab of half thickness :math:`a` with the condition :math:`-D
  \nabla\phi \cdot \mathbf{n} = \phi / r_v`, for :math:`r_v = 2` and
  :math:`r_v = 2.1312`, the flux is :math:`\cos(B x)` with :math:`D B
  \tan(B a) = 1/r_v`, where :math:`x` is the position from the middle of the
  slab and :math:`B` is in :math:`\mathrm{1/m}`.  The test checks
  :math:`k = \nu\Sigma_f / (\Sigma_a + D B^2)` to a relative
  :math:`10^{-6}`.
* In an infinite medium of the fuel of the IAEA benchmark, :math:`k_\infty =
  1.125` ([Lee2025]_, Eq. 7.27) to round-off.
* The IAEA PWR benchmark gives :math:`k_\mathrm{eff}` within 6 pcm of the
  reference 1.02959 with 8 finite elements per assembly, and the order of
  convergence of its error exceeds 1.8.
