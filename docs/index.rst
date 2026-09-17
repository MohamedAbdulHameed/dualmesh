dualmesh
========

**dualmesh** solves boundary value problems of heat transfer, solid mechanics,
structural mechanics, and viscous incompressible flow with the *dual mesh
control domain method* (DMCDM) of J. N. Reddy, and — using exactly the same
problem definition — with the standard Galerkin finite element method, so that
the two can be compared directly.

The library is written in C++17 and driven from Python.  Its structure follows
the `MOOSE framework <https://mooseframework.inl.gov>`_: physics is added as
*kernels*, *boundary conditions*, and *materials*, which are registered objects
with validated, self-documenting parameters; meshes are either generated or
read from standard files; and the solvers (Newton with exact automatic
differentiation, direct iteration, load stepping, time integration) are shared
by every physics module.

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05,
       num_x_elements=10, num_y_elements=5)

   problem = dm.Problem(mesh, method="dmcdm")
   problem.add_variable("temperature")
   problem.add_kernel("HeatConduction", variable="temperature",
                      thermal_conductivity=20.0)
   problem.add_kernel("HeatSource", variable="temperature", heat_source=1.0e6)
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="left", value=40.0)
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="right", value=10.0)
   problem.add_boundary_condition("ConvectiveHeatFluxBC", variable="temperature",
                                  boundary="top", heat_transfer_coefficient=75.0)
   problem.solve()

   print(problem.sample("temperature", [[0.05, 0.0]]))   # 83.142 (Reddy, Table 5.4.3)

What the method is
------------------

The dual mesh control domain method keeps two meshes.  The **primal mesh** is a
mesh of finite elements and provides the interpolation of the unknowns.  The
**dual mesh** is the set of node-centred *control domains* obtained by joining
edge midpoints, face centroids, and element centroids.  The governing equation
is then satisfied in the *integral* sense over each control domain, without a
weight function:

.. math::

   \int_{CD_I} \left[ -\nabla \cdot \mathbf{F} + S \right] \, dV = 0
   \quad \Longrightarrow \quad
   -\oint_{\partial CD_I} \mathbf{F} \cdot \mathbf{n} \, dS
   + \int_{CD_I} S \, dV = 0 .

The surface integral makes the secondary variables (fluxes, forces, moments)
appear naturally on the control domain interfaces, which is the physical
appeal of the finite volume method, while the primal interpolation removes the
ad-hoc gradient reconstructions that the finite volume method needs.  Chapter 5
of Reddy's book develops the method; :doc:`theory` summarizes it in the form
the code implements.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   theory
   tutorials/index
   input_files
   objects
   api
   verification
   openfoam
   developing
   references

Citing
------

If this software contributes to your work, please cite both the method and the
software; see :doc:`references` and the ``CITATION.cff`` file in the
repository.
