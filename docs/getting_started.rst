Getting started
===============

This page walks through a complete problem and then explains each ingredient.
The problem is Example 5.4.3 of the book: a bus bar of 0.10 m by 0.05 m
carrying enough current to generate :math:`10^6` W/m³, held at 40 °C on the
left and 10 °C on the right, insulated at the bottom, and cooled by air at
0 °C on top with a film coefficient of 75 W/(m² K).

.. code-block:: python

   import dualmesh as dm

   # 1. a mesh
   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05,
       num_x_elements=10, num_y_elements=5)

   # 2. a problem on that mesh
   problem = dm.Problem(mesh, method="dmcdm")

   # 3. the unknown field
   problem.add_variable("temperature")

   # 4. the terms of the equation
   problem.add_kernel("HeatConduction", variable="temperature",
                      thermal_conductivity=20.0)
   problem.add_kernel("HeatSource", variable="temperature", heat_source=1.0e6)

   # 5. the boundary conditions
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="left", value=40.0)
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="right", value=10.0)
   problem.add_boundary_condition("ConvectiveHeatFluxBC", variable="temperature",
                                  boundary="top", heat_transfer_coefficient=75.0,
                                  ambient_temperature=0.0)
   # the bottom is insulated: a zero natural condition, so nothing to add

   # 6. solve
   problem.solve()

   # 7. results
   print(problem.sample("temperature", [[0.05, 0.0], [0.05, 0.05]]))
   print(problem.total_reaction("temperature", "left"))
   problem.write_vtu("bus_bar.vtu")

The printed temperatures are 83.142 °C and 76.859 °C, which are the values of
Table 5.4.3 of the book to every printed digit.

Meshes
------

Generate one:

.. code-block:: python

   dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
   dm.generate_rectangle_mesh(x_min=0, x_max=2, y_min=0, y_max=1,
                              num_x_elements=20, num_y_elements=10,
                              element_type="Tri3")      # or "Quad4"
   dm.generate_box_mesh(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1,
                        num_x_elements=8, num_y_elements=8, num_z_elements=8)
   dm.generate_annulus_mesh(inner_radius=0.05, outer_radius=0.1,
                            num_radial_elements=8, num_angular_elements=19)

Non-uniform spacing is available either as a bias (the ratio between successive
element lengths) or as explicit coordinates:

.. code-block:: python

   dm.generate_rectangle_mesh(x_coordinates=[0, 0.5, 1.5, 3.0],
                              y_coordinates=dm.graded_coordinates(0, 1, 10, bias=0.8))

Or read one written by another tool — Gmsh, Exodus, VTK, Abaqus, and everything
else `meshio <https://github.com/nschloe/meshio>`_ supports:

.. code-block:: python

   mesh = dm.read_mesh("channel.msh")           # physical groups become side sets
   mesh = dm.read_mesh("bracket.e")

Side sets and node sets carry the names used in the boundary conditions.  The
generators create ``left``, ``right``, ``bottom``, ``top``, ``back``, ``front``
(and ``inner``, ``outer``, ``start``, ``end`` for an annulus); a file's physical
groups keep their names.  More can be added geometrically:

.. code-block:: python

   mesh.add_nodeset_by_predicate("centre", lambda x, y, z: abs(x) < 1e-12)
   mesh.add_sideset_by_predicate("hot_wall", lambda x, y, z: y > 0.999)

Variables, kernels, boundary conditions, materials
--------------------------------------------------

A **variable** is a nodal unknown.  A **kernel** contributes a flux, a source,
or both to the equation of one variable.  A **boundary condition** either
prescribes the variable (``DirichletBC`` and friends) or prescribes the normal
flux (``NeumannBC``, ``RobinBC``, ``ConvectiveHeatFluxBC``, ``TractionBC``,
``PressureBC``, ...).  A **material** computes named properties, such as
``"stress"``, that kernels then use.

Every object is created by its registered name and validated against its
declared parameters, so a typo is reported with the list of accepted
parameters.  To see what exists:

.. code-block:: python

   dm.list_objects(category="Kernel")
   print(dm.describe("HeatConduction"))

or at the command line, ``dualmesh list`` and ``dualmesh describe
HeatConduction``.

Coefficients can be numbers, named functions, or plain Python callables
``f(x, y, z, t)``:

.. code-block:: python

   problem.add_boundary_condition(
       "DirichletBC", variable="temperature", boundary="top",
       value=lambda x, y, z, t: 500.0 * (1.0 - 10.0 * x * x))

Solving
-------

.. code-block:: python

   problem.solve()                                   # steady, Newton
   problem.solve(nonlinear_solver="picard", relaxation=0.35)
   problem.solve(load_factors=[0.25, 0.5, 0.75, 1.0])   # load stepping
   problem.solve_transient(end_time=10.0, dt=0.5, theta=1.0)

Results
-------

.. code-block:: python

   problem.values("temperature")                     # one value per node
   problem.sample("temperature", [[0.05, 0.0]])      # anywhere in the mesh
   problem.gradient_at_centroids("temperature")
   problem.property_at_centroids("stress")
   problem.reactions("temperature", "left")          # node by node
   problem.total_reaction("temperature", "left")     # summed
   problem.integrate("temperature")
   problem.write_vtu("out.vtu", cell_properties=["stress"])
   problem.write_csv("out.csv")
   problem.write_mesh_file("out.e")                  # through meshio

Comparing the two methods
-------------------------

The only change needed to run the same problem with the Galerkin finite element
method is the ``method`` argument:

.. code-block:: python

   dual = dm.Problem(mesh, method="dmcdm")
   finite_element = dm.Problem(mesh, method="fem")
