Case files
==========

A case file holds the input of a problem at the physics level as data, in
YAML.  It has the same names as the Python input: the keywords of the mesh
generators, the types and parameters of the physics and couplings, the
boundary conditions written as text, and the options of the solve.  A case file
therefore needs no programming, and every case file has a Python script that
builds exactly the same problem.  The object level (kernels, property objects
and boundary condition objects added one by one) stays a Python tool.

Run a case file
---------------

Run a case file from the command line:

.. code-block:: bash

   dualmesh run heated_plate.yaml
   mpirun -n 4 dualmesh run heated_plate.yaml

In Python, :func:`dualmesh.run_case` runs a case file or a dict of the same
sections and returns a :class:`~dualmesh.Case`, whose ``problem`` holds the
solved problem, ``physics`` the physics by their keys and ``result`` the result
of the study.

An example
----------

A steel plate is heated inside, held at 300 K on its left edge, cooled by air
on its top edge and clamped on the left, so that it expands as it heats:

.. literalinclude:: ../../examples/heated_plate.yaml
   :language: yaml

The same problem in Python is:

.. code-block:: python

   mesh = dm.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[40, 8])
   problem = dm.Problem(mesh, method="fem")
   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("temperature = 300", boundary="left")
   heat.add_boundary_condition("heat_flux = h*(temperature - 293)", boundary="top", h=25.0)
   solid = problem.add_physics(
       "solid_mechanics",
       formulation="plane_stress",
       youngs_modulus=2.0e11,
       poissons_ratio=0.3,
       thickness=0.01,
   )
   solid.add_boundary_condition("fixed", boundary="left")
   problem.add_coupling(
       "thermal_expansion",
       heat_transfer=heat,
       solid_mechanics=solid,
       linear_expansion_coeff=1.2e-5,
       stress_free_temperature=293,
   )
   problem.solve(report="summary")
   problem.write_vtu("heated_plate.vtu")
   problem.write_csv("heated_plate.csv")

The test suite runs both and checks that their fields are identical, and that
the 2000 W/m generated in the plate leaves through the left and the top edges.

The sections
------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Section
     - Content
   * - ``mesh``
     - ``generate:`` the name of a generator (``rectangle`` for
       :func:`~dualmesh.generate_rectangle_mesh`, ``box``, ``line``,
       ``annulus``, ``cylinder_surface``, ``spherical_shell``, ``surface``)
       with its keywords, or ``file:`` a mesh file with the keywords of
       :func:`~dualmesh.read_mesh`.  Required.
   * - ``method``
     - ``fem``, ``dmcdm``, ``hfvm`` or ``zfvm``.  Default ``dmcdm``.
   * - ``coordinates``
     - ``cartesian`` (the default), ``axisymmetric`` or ``spherical``.
   * - ``physics``
     - One entry per physics, under the name a script gives its variable
       (``heat``, ``solid``), with its ``type``, its parameters and a list of
       ``conditions``.  A condition has its ``boundary`` and its ``condition``,
       the text of :meth:`~dualmesh.physics.Physics.add_boundary_condition`,
       and every other key is a definition that the text uses by its name.
       Required.
   * - ``couplings``
     - A list of couplings, each with its ``type`` and parameters.  A
       parameter that takes a physics takes its key.
   * - ``solve``
     - ``study``: ``steady`` (the default), ``transient`` or ``eigenvalue``,
       with the options of :meth:`~dualmesh.Problem.solve`,
       :meth:`~dualmesh.Problem.solve_transient` or
       :meth:`~dualmesh.Problem.solve_eigenvalue`.
   * - ``output``
     - ``fields``: a VTU file of the fields, and ``csv``: a CSV file of the
       values at the unknowns.

Every name is checked before any work is done: an unknown section, generator,
study or output, or a parameter that a physics does not have, is refused with
the closest valid name.  Numbers follow YAML 1.2, so ``1.0e6`` is a number, as
it is in Python.
