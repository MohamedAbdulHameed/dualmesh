Quickstart
==========

This page solves one heat conduction problem twice: from a case file with the
command ``dualmesh run``, and from Python with :class:`dualmesh.problem`.  It
assumes that you installed the package as :doc:`installation` describes.

The problem
-----------

A slab of thickness :math:`L = 0.1\ \mathrm{m}` and thermal conductivity
:math:`k = 45\ \mathrm{W/(m\,K)}` separates a hot wall from air.  Its left face,
at :math:`x = 0`, is held at the temperature
:math:`T_\mathrm{h} = 400\ \mathrm{K}`.  Its right face, at :math:`x = L`,
loses heat by convection to air at the temperature
:math:`T_\mathrm{a} = 300\ \mathrm{K}`, with the heat transfer coefficient
:math:`h_\mathrm{c} = 250\ \mathrm{W/(m^2\,K)}`.

In the steady state, the conduction through the slab and the convection to the
air act as two thermal resistances per unit area in series, :math:`L/k` and
:math:`1/h_\mathrm{c}`, both in :math:`\mathrm{m^2\,K/W}`.  The heat flux :math:`\mathbf{q}` through the right
face, along its outward unit normal :math:`\mathbf{n}`, is

.. math::

   \mathbf{q} \cdot \mathbf{n}
   = \frac{T_\mathrm{h} - T_\mathrm{a}}{L/k + 1/h_\mathrm{c}}
   = 16\,071.4\ \mathrm{W/m^2},

and the right face is at the temperature

.. math::

   T(L) = T_\mathrm{h} - \frac{(\mathbf{q} \cdot \mathbf{n})\, L}{k}
   = 364.29\ \mathrm{K}.

The temperature is linear in :math:`x`, which linear elements represent
exactly, so the computed values agree with these values to eight significant
digits or more.  The test
``test_a_slab_cooled_by_convection_has_the_heat_flow_of_its_series_resistance``
in ``tests/python/test_front_end.py`` checks both.

Solve from a case file
----------------------

1. Save the case as ``slab.yaml``:

   .. code-block:: yaml

      mesh:
        segment:
          length: 0.1 m
          cells: 10
      physics:
        heat_conduction: {}
      properties:
        thermal_conductivity: 45 W/(m*K)
      boundary_conditions:
        left:
          temperature: 400 K
        right:
          convection:
            heat_transfer_coefficient: 250 W/(m^2*K)
            ambient_temperature: 300 K
      postprocessors:
        heat_to_air:
          boundary_flux: heat_flux
          boundary: right
      output:
        file: slab
        fields: [temperature]

2. Run the case:

   .. code-block:: console

      dualmesh run slab.yaml

Each section of the file sets one part of the problem:

* ``mesh`` generates a segment of 10 linear cells from :math:`x = 0` to
  :math:`x = L`.  Its two ends are the boundaries ``left`` and ``right``.
* ``physics`` adds ``heat_conduction``, which solves for the field
  ``temperature``.  The empty mapping ``{}`` gives it no options.
* ``properties`` gives the thermal conductivity by its standard name, with its
  unit.
* ``boundary_conditions`` holds the temperature of ``left`` and the
  convection of ``right``, with its two constants.
* ``postprocessors`` defines ``heat_to_air``, the heat flux that leaves through
  ``right``.
* ``output`` writes the temperature and the post-processors to files that
  start with ``slab``.

The command prints a line that names the solve, its number of unknowns and its
method, then the result of the solve.  It writes the files in the folder of
``slab.yaml``:

* ``slab.vtu``, the temperature on the mesh, in the unstructured grid
  format of the Visualization Toolkit (VTK), which ParaView reads. A transient
  run writes ``slab_00000.vtu``, ``slab_00001.vtu`` and so on, with the
  collection ``slab.pvd``, which ParaView opens as one time series.
* ``slab.csv``, the post-processors as comma-separated values (CSV).  Its
  header is ``time [s],heat_to_air [W/m^2]``, and a steady run gives one row,
  at the time 0 s.

``dualmesh run`` reads a TOML file of the same sections in the same way.

Solve from Python
-----------------

The same case in Python is one call per section:

.. code-block:: python

   import dualmesh
   import numpy

   slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}})
   slab.physics("heat_conduction")
   slab.properties(thermal_conductivity="45 W/(m*K)")
   slab.boundary_conditions("left", temperature="400 K")
   slab.boundary_conditions(
       "right",
       convection={"heat_transfer_coefficient": "250 W/(m^2*K)", "ambient_temperature": "300 K"},
   )
   slab.postprocessor("heat_to_air", boundary_flux="heat_flux", boundary="right")
   slab.output(file="slab", fields=["temperature"])
   solution = slab.solve()

   flux = solution.postprocessor("heat_to_air")[-1]
   print(f"heat_to_air: {flux:.1f} {solution.unit('heat_to_air')}")
   x = solution.coordinates("temperature")[:, 0]
   temperature = solution.field("temperature")[:, 0]
   print(f"cooled face: {temperature[numpy.argmax(x)]:.2f} K")

The script writes the same files as the case file, in the working folder, and
prints:

.. code-block:: text

   heat_to_air: 16071.4 W/m^2
   cooled face: 364.29 K

Each call checks its own section, so a misspelled key or a wrong unit fails at
the line that gives it.  :meth:`~dualmesh.problem.solve` returns a
:class:`dualmesh.results`, which holds the fields by name, with the
coordinates of their nodes in metres, and the post-processors in the units of
the International System of Units (SI).
``solve(report="none")`` solves without printing.

The two forms build the same input.  ``slab.to_YAML("slab.yaml")`` writes the
problem as a case file, and ``dualmesh.problem.from_file("slab.yaml")`` reads a
case file into a problem that you can change and solve from Python.

Next steps
----------

* :doc:`concepts` defines the parts of a problem and the two methods.
* :doc:`user_guide/input` lists every section and key of the input.
* :doc:`tutorials/index` solves problems of heat transfer, solid mechanics and
  flow.
