.. _input:

The input
=========

The input describes a whole simulation as data: the mesh, the physics, the
material properties, the initial and boundary conditions, the time span, the
mesh adaptivity, the quantities to report and the solver settings. You write
it in Python with :class:`dualmesh.problem`, or as a case file in YAML (YAML
Ain't Markup Language) or TOML (Tom's Obvious Minimal Language). All three
build the same input tree, and the same checks apply to each.

This page is the reference of that tree. It first shows the three ways to write
an input and how to run it. It then describes the values that every section
uses: quantities with units, the standard names, formulas, tables and regions.
The last part takes the top-level sections in the order of the tree and gives,
for each key, its meaning, its unit, its allowed values and its default.

.. contents::
   :local:
   :depth: 2

.. _input-write-and-run:

Write and run an input
----------------------

The example is a slab of thickness :math:`L = 0.1\,\mathrm{m}` with a thermal
conductivity :math:`k = 45\,\mathrm{W/(m\,K)}` and a uniform volumetric heat
source :math:`s = 1\,\mathrm{MW/m^3}`. Both faces stay at
:math:`300\,\mathrm{K}`. The test ``tests/python/test_front_end.py`` solves
this slab and checks it against the exact temperature
:math:`T(x) = 300\,\mathrm{K} + s\,x\,(L - x)/(2k)`, with :math:`x` the
position across the slab in m.

In Python
~~~~~~~~~

Each method of :class:`dualmesh.problem` sets one section of the tree:

.. code-block:: python

   import dualmesh

   slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 20}})
   slab.physics("heat_conduction")
   slab.properties(thermal_conductivity="45 W/(m*K)", volumetric_heat_source="1 MW/m^3")
   slab.boundary_conditions("left", temperature="300 K")
   slab.boundary_conditions("right", temperature="300 K")
   solution = slab.solve()
   temperature = solution.field("temperature")[:, 0]
   x = solution.coordinates("temperature")[:, 0]

:meth:`~dualmesh.problem.solve` returns a :class:`dualmesh.results` object
with the fields, their node coordinates in m and the postprocessors.
Its argument ``report`` chooses what the solve prints: ``"summary"`` (the
default), ``"full"`` or ``"none"``. :doc:`/user_guide/output` describes the
results.

.. list-table:: The methods of :class:`dualmesh.problem` and the sections they set
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Python
   * - ``mesh``, ``method``
     - ``dualmesh.problem(mesh={...}, method=None)``
   * - ``physics``
     - ``physics(name, **options)``, once per physics
   * - ``properties``
     - ``properties(**values)``
   * - ``gravity``
     - ``gravity([...])``
   * - ``models``
     - ``model(name, **entry)``, once per model
   * - ``initial_conditions``
     - ``initial_conditions(**values)``
   * - ``boundary_conditions``
     - ``boundary_conditions(boundary, **values)``, once per boundary, or
       ``boundary_conditions(boundary, "symmetry")``
   * - ``time``
     - ``time(end_time=..., initial_time_step=..., step_control={...})``
   * - ``eigenvalues``
     - ``eigenvalues(count=..., near=...)``
   * - ``mesh_adaptivity``
     - ``mesh_adaptivity(**values)``
   * - ``postprocessors``
     - ``postprocessor(name, **definition)``, once per postprocessor. The key
       ``from`` of a line is ``from_``, since ``from`` is a word of Python.
   * - ``output``
     - ``output(file, fields, interval)``
   * - ``solver``
     - ``solver(**values)``

A second call of ``physics``, ``properties``, ``initial_conditions``,
``postprocessor`` or ``solver`` adds its keys to those of the first, and a key
given again replaces its earlier value. A second call of
``boundary_conditions`` for the same boundary adds its keys to that boundary.
A second call of ``gravity``, ``time``, ``eigenvalues``, ``mesh_adaptivity``
or ``output`` replaces the whole section.

Each call checks its own section at once, so a wrong key stops the script at
the line that gives it, and the problem keeps its earlier state. The checks
that join several sections, such as a property that a physics needs, run in
:meth:`~dualmesh.problem.solve`.

In a case file
~~~~~~~~~~~~~~

A case file holds the same tree, with one top-level key per section. In YAML
the slab reads:

.. code-block:: yaml

   mesh:
     segment:
       length: 0.1 m
       cells: 20
   physics:
     heat_conduction: {}
   properties:
     thermal_conductivity: 45 W/(m*K)
     volumetric_heat_source: 1 MW/m^3
   boundary_conditions:
     left:
       temperature: 300 K
     right:
       temperature: 300 K

In TOML it reads:

.. code-block:: toml

   [mesh.segment]
   length = "0.1 m"
   cells = 20

   [physics.heat_conduction]

   [properties]
   thermal_conductivity = "45 W/(m*K)"
   volumetric_heat_source = "1 MW/m^3"

   [boundary_conditions.left]
   temperature = "300 K"

   [boundary_conditions.right]
   temperature = "300 K"

A physics without options is an empty mapping: ``{}`` in YAML and a table
header alone in TOML.

The file name ends in ``.yaml``, ``.yml`` or ``.toml``. The file holds one
mapping of sections. The input reads YAML by the rules of YAML 1.2:
``true`` and ``false`` are the only Booleans, ``1e-3`` is a number, and a word
such as ``yes`` or a number such as ``010`` stays a text. In YAML, put every
formula in quotes, because an unquoted text that starts with ``[`` reads as a
list.

Run a case file
~~~~~~~~~~~~~~~

Run a case file from the command line:

.. code-block:: console

   dualmesh run case.yaml
   mpirun -n 4 dualmesh run case.yaml

The second command runs the case on four processes with MPI (the Message
Passing Interface).

The file ``examples/heated_plate.yaml`` is a case file of a steel plate that
is heated inside, held at :math:`300\,\mathrm{K}` and clamped on its left
side, and cooled by air on its top side. It couples ``heat_conduction`` with
``solid_mechanics`` through a thermal expansion:

.. literalinclude:: ../../examples/heated_plate.yaml
   :language: yaml

The test ``tests/python/test_front_end.py`` runs it and checks that the
:math:`2000\,\mathrm{W/m}` generated in the plate, per unit depth, leaves
through the left and the top sides.

In Python, :meth:`dualmesh.problem.from_file` reads a case file and returns a
problem:

.. code-block:: python

   import dualmesh

   solution = dualmesh.problem.from_file("case.yaml").solve()

A problem writes its tree to a file with ``to_YAML(path)`` or
``to_TOML(path)``, and returns it as a Python dictionary with ``to_tree()``.
The file holds the values as you wrote them, with their units, in the order of
the sections. The test ``tests/python/test_front_end.py`` writes the slab with
``to_YAML``, reads it back with ``from_file`` and checks that both give the
same temperatures.

A relative path in a case file, for a mesh file or a table, resolves against
the folder of the case file. A relative path in Python resolves against the
working directory. ``to_YAML`` and ``to_TOML`` rewrite each relative path so
that it resolves against the folder of the file they write.

Messages
~~~~~~~~

Every key is checked before the solve starts. A wrong input raises
``dualmesh.InputError`` with a message that starts with the path of the key,
such as ``properties.thermal_conductivity``. A misspelled key or name comes
with the nearest valid one.

.. _input-values:

Values
------

.. _input-units:

Quantities and units
~~~~~~~~~~~~~~~~~~~~

A quantity is a text that holds a number, a space and a unit, such as
``"7850 kg/m^3"`` or ``"200 GPa"``. The input converts every quantity to the
International System of Units (SI) when it reads it, and checks its dimension
against that of its key. A bare number, such as ``0.3`` or ``"0.3"``, is
dimensionless. A key with a dimension refuses a bare number, and the message
shows the value with the unit of the key.

A unit is a product of unit symbols:

- ``*`` multiplies and ``/`` divides, from left to right, so ``J/mol/K`` is
  ``J/(mol*K)``.
- ``^`` raises a symbol or a parenthesis to an integer power, such as ``m^3``
  or ``s^-1``, or to a fraction in parentheses, such as ``m^(1/2)``.
- ``1`` is the unit of a dimensionless quantity, as in ``1/K``.
- Spaces between the parts of a unit are allowed.

.. list-table:: The unit symbols
   :header-rows: 1
   :widths: 30 70

   * - Kind
     - Symbols
   * - SI base units
     - ``m``, ``kg``, ``g``, ``s``, ``A``, ``K``, ``mol``, ``cd``
   * - SI derived units
     - ``rad``, ``sr``, ``Hz``, ``N``, ``Pa``, ``J``, ``W``, ``C``, ``V``,
       ``F``, ``ohm``, ``S``, ``Wb``, ``T``, ``H``, ``lm``, ``lx``, ``Bq``,
       ``Gy``, ``Sv``, ``kat``
   * - Other units
     - ``deg`` (the degree of angle), ``L`` (the litre), ``min``, ``h``,
       ``d`` (the day), ``bar``, ``eV``
   * - Temperature scales
     - ``degC``, ``degF``

A symbol takes one of the SI prefixes ``p``, ``n``, ``u`` (micro), ``m``,
``c``, ``d``, ``da``, ``h``, ``k``, ``M`` and ``G``, as in ``mm``, ``kPa`` or
``MW``. The symbols ``kg``, ``deg``, ``min``, ``h`` and ``d`` take no prefix.
A symbol of the table wins over a prefixed reading, so ``min`` is the minute
and ``cd`` the candela.

The scales ``degC`` and ``degF`` have an offset from the kelvin. They give a
temperature only alone, as in ``"40 degC"``, which reads as
:math:`313.15\,\mathrm{K}`. A temperature difference and a compound unit
use ``K``, as in ``W/(m*K)``.

.. _input-standard-names:

Standard names
~~~~~~~~~~~~~~

Every field, property and boundary quantity has a standard name with a fixed
unit and shape. The input takes the unit of a name from this vocabulary, so a
property is written ``density: 7850 kg/m^3`` without a separate unit key. The
vocabulary is the file ``python/dualmesh/standard_names.toml``.

.. list-table:: The standard names
   :header-rows: 1
   :widths: 34 20 14 32

   * - Name
     - SI unit
     - Shape
     - Meaning
   * - ``temperature``
     - ``K``
     - scalar
     - thermodynamic temperature
   * - ``displacement``
     - ``m``
     - vector
     - displacement of a material point
   * - ``velocity``
     - ``m/s``
     - vector
     - flow velocity
   * - ``pressure``
     - ``Pa``
     - scalar
     - pressure of a fluid
   * - ``concentration``
     - ``mol/m^3``
     - scalar
     - amount of a species per volume
   * - ``damage``
     - ``1``
     - scalar
     - crack indicator from 0 (intact) to 1 (broken)
   * - ``heat_flux``
     - ``W/m^2``
     - vector
     - heat flow per area
   * - ``stress``
     - ``Pa``
     - tensor
     - Cauchy stress
   * - ``strain``
     - ``1``
     - tensor
     - small or logarithmic strain
   * - ``von_Mises_stress``
     - ``Pa``
     - scalar
     - equivalent stress of the stress deviator
   * - ``plastic_strain``
     - ``1``
     - tensor
     - plastic part of the strain
   * - ``equivalent_plastic_strain``
     - ``1``
     - scalar
     - accumulated plastic strain
   * - ``density``
     - ``kg/m^3``
     - scalar
     - mass per volume
   * - ``specific_heat_capacity``
     - ``J/(kg*K)``
     - scalar
     - heat capacity per mass at constant pressure
   * - ``thermal_conductivity``
     - ``W/(m*K)``
     - scalar or tensor
     - coefficient of Fourier's law
   * - ``linear_expansion_coefficient``
     - ``1/K``
     - scalar
     - relative change of length per change of temperature
   * - ``volumetric_expansion_coefficient``
     - ``1/K``
     - scalar
     - relative change of volume per change of temperature
   * - ``reference_temperature``
     - ``K``
     - scalar
     - temperature at which the density of a heated fluid holds
   * - ``stress_free_temperature``
     - ``K``
     - scalar
     - temperature of zero thermal strain
   * - ``Young_modulus``
     - ``Pa``
     - scalar
     - uniaxial elastic modulus
   * - ``Poisson_ratio``
     - ``1``
     - scalar
     - lateral contraction per axial extension
   * - ``shear_modulus``
     - ``Pa``
     - scalar
     - elastic modulus in shear
   * - ``yield_stress``
     - ``Pa``
     - scalar
     - equivalent stress at the onset of plastic flow
   * - ``dynamic_viscosity``
     - ``Pa*s``
     - scalar
     - Newtonian viscosity
   * - ``diffusivity``
     - ``m^2/s``
     - scalar or tensor
     - coefficient of Fick's law
   * - ``burnup``
     - ``J/kg``
     - scalar
     - fission energy released per mass of initial heavy metal
   * - ``volumetric_heat_source``
     - ``W/m^3``
     - scalar
     - heat produced per volume and time
   * - ``body_force_density``
     - ``N/m^3``
     - vector
     - force per volume
   * - ``traction``
     - ``Pa``
     - vector
     - force per area on a surface
   * - ``heat_transfer_coefficient``
     - ``W/(m^2*K)``
     - scalar
     - convective heat flux per temperature difference
   * - ``ambient_temperature``
     - ``K``
     - scalar
     - temperature of the surroundings
   * - ``emissivity``
     - ``1``
     - scalar
     - radiative emissivity of a surface
   * - ``thickness``
     - ``m``
     - scalar
     - out-of-plane thickness of a plane problem or a shell
   * - ``time``
     - ``s``
     - scalar
     - time
   * - ``time_step``
     - ``s``
     - scalar
     - step of a transient run

A component of a vector takes the suffix of its direction, as in
``displacement_x``, ``velocity_y`` or ``heat_flux_z``. A component of a tensor
takes two suffixes, as in ``stress_xy``. In axisymmetric coordinates the
suffixes are ``r`` for the radius and ``z`` for the axis, as in
``displacement_r`` and ``stress_rz``.

The vocabulary also holds five physical constants, which a formula reads by
name and no input can set: ``Boltzmann_constant``, ``Avogadro_constant``,
``elementary_charge``, ``molar_gas_constant`` and
``Stefan_Boltzmann_constant``. Their values are those of the table of the
National Institute of Standards and Technology (NIST).

.. _input-formulas:

Formulas
~~~~~~~~

A property or a boundary value can be a formula: a text that computes the
value from the fields, the position and the time. The conductivity of a
material that falls with the temperature reads:

.. code-block:: yaml

   properties:
     thermal_conductivity: "1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)"

You write formulas in one language everywhere: in the input, in the models of
your own and in the equations of ``general_form_PDE``
(:doc:`/user_guide/equations`). A formula holds these parts:

- A quantity in square brackets, a number and its unit such as
  ``[0.0375 m*K/W]`` or ``[2.77 eV]``, is a number converted to SI. Every
  quantity of a formula takes its brackets, so a constant plus a term reads
  ``[300 K] + [50 K/m]*x``.
- A bare number, such as ``2`` or ``1.5e-3``, is dimensionless.
- Square brackets around anything else, such as ``[1, 0]``, hold the items of
  a vector, one per direction of space. A list of vectors holds the rows of a
  tensor.
- The operators are ``+``, ``-``, ``*``, ``/`` and ``^``, or ``**`` for the
  power. A product needs ``*``. The power binds tightest and groups from the
  right, so ``2^3^2`` is :math:`2^9`, and a unary minus binds less tightly than
  the power, so ``-a^2`` is :math:`-(a^2)`. Parentheses group as usual.
- The comparisons ``<``, ``>``, ``<=``, ``>=``, ``==`` and ``!=`` give 1 where
  they hold and 0 elsewhere. ``and`` and ``or`` join two conditions, each of
  which holds where it is not 0.
- A field by its standard name, such as ``temperature`` or ``pressure``, or a
  component of one, such as ``displacement_x``.
- The position in m: ``x``, ``y`` and ``z`` in Cartesian coordinates, and
  ``r`` and ``z`` in axisymmetric coordinates.
- ``time``, the time in s.
- A physical constant by name, such as ``Stefan_Boltzmann_constant``.

A formula of a property or of a boundary value calls these functions:

.. list-table:: The functions of a formula
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Value
   * - ``exp``, ``log``, ``log10``, ``sin``, ``cos``, ``tan``, ``asin``,
       ``acos``, ``atan``, ``sinh``, ``cosh``, ``tanh``, ``erf``, ``floor``,
       ``ceil``
     - the function of one dimensionless argument, with ``log`` the natural
       logarithm and ``log10`` the logarithm to base 10
   * - ``sqrt(a)``, ``abs(a)``, ``sign(a)``
     - the square root, the absolute value, and 1, 0 or -1 by the sign of
       ``a``
   * - ``max(a, b)``, ``min(a, b)``, ``hypot(a, b)``
     - the larger and the smaller of two values of the same dimension, and
       :math:`\sqrt{a^2 + b^2}`
   * - ``atan2(a, b)``, ``pow(a, b)``
     - the angle of the point :math:`(b, a)` in rad, and :math:`a^b`
   * - ``less(a, b)``, ``greater(a, b)``
     - the conditions :math:`a < b` and :math:`a > b`, for two values of
       the same dimension
   * - ``conditional(c, a, b)``
     - ``a`` where the condition ``c`` holds and ``b`` elsewhere, with ``a``
       and ``b`` of the same dimension

The input checks the dimension of every formula against the unit of its key.
A sum, ``max``, ``min``, ``hypot``, ``atan2``, a comparison and the two
branches of ``conditional`` need operands of the same dimension. A power of a
quantity with a dimension needs a number as its exponent. The number ``0``
matches every dimension. The message of a formula of the wrong dimension names
the unit that the formula gives.

The formula language also parses ``time_step``, ``cell_size``, ``grad`` and
the tensor functions. The solve refuses them in a property or a boundary
value, which is a function of the fields, the position and the time.

.. _input-tables:

Tables
~~~~~~

A table gives a property as its values at the rows of one input, such as the
temperature, and the solve interpolates linearly between the rows. It is a
mapping with the one key ``table``, which holds the columns inline or names a
file of comma-separated values (CSV):

.. code-block:: yaml

   properties:
     Young_modulus:
       table:
         temperature: {unit: K, values: [300, 600]}
         Young_modulus: {unit: GPa, values: [96, 83]}

.. code-block:: yaml

   properties:
     thermal_conductivity:
       table: {file: zircaloy_conductivity.csv}

The first line of a CSV file names each column with its unit in square
brackets, as in ``temperature [K], thermal_conductivity [W/(m*K)]``. A
dimensionless column may leave out the brackets. Each further line holds one
number per column. Blank lines are skipped. In Python, :func:`dualmesh.table`
builds both forms:

.. code-block:: python

   import dualmesh

   bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}})
   bar.properties(
       Young_modulus=dualmesh.table(temperature=("K", [300, 600]), Young_modulus=("GPa", [96, 83]))
   )
   # The file zircaloy_conductivity.csv sits in the working directory.
   bar.properties(thermal_conductivity=dualmesh.table("zircaloy_conductivity.csv"))

The input checks a table against these rules:

- The output column has the name of the property. The other column is the
  input, a field, the time or a coordinate, such as ``temperature``, ``time``
  or ``x``.
- The table has one input.
- Every column holds the same number of values.
- The unit of each column has the dimension of its name.
- The input holds at least two values, and no value twice. The rows may come
  in any order.

Between two rows, the property is the straight line through them. The solve
takes the derivative of that line with respect to the input, so Newton's
method converges on a table of a field as on a formula. A table read outside
its range stops the run with a message that names the property, the region and
the value, for example:

.. code-block:: text

   the table of the property thermal_conductivity in the region cladding reads
   temperature = 1300, outside the range of the table, from 300 to 1200.

The solve first tries smaller steps, since a Newton iterate can leave the range
on the way to a solution inside it. The properties that a physics reads as a
field, such as ``thermal_conductivity`` and ``Young_modulus``, take a table.
The properties that a physics takes as one number, such as the
``dynamic_viscosity`` of a flow, take a constant, and a boundary condition
takes a constant or a formula.

.. _input-regions:

Regions
~~~~~~~

A property can take one value per region of the mesh. The value is then a
mapping of region names to values, here for a mesh file with the regions
``fuel`` and ``cladding``:

.. code-block:: yaml

   properties:
     thermal_conductivity:
       fuel: 3 W/(m*K)
       cladding: 16 W/(m*K)

A generated mesh has the one region ``domain``. The regions of a mesh file are
its named element blocks, which the solve checks. A region cannot take the
name ``table``, ``unit`` or ``value``. :ref:`input-properties` lists the
properties that take a value per region.

.. _input-sections:

The sections
------------

The tree holds these top-level sections, in this order: ``mesh``, ``method``,
``physics``, ``properties``, ``gravity``, ``models``, ``initial_conditions``,
``boundary_conditions``, ``time``, ``eigenvalues``, ``mesh_adaptivity``,
``postprocessors``, ``output`` and ``solver``. Only ``mesh`` is required by the check of the
tree, and the solve also needs ``physics``.

.. _input-mesh:

mesh
~~~~

The section ``mesh`` holds exactly one of ``segment``, ``rectangle``, ``box``
and ``file``, and the optional key ``coordinate_system``.

A generated mesh has linear cells. Its one region is ``domain``. Each form
takes two optional keys besides those of the table:

``origin``
   The corner with the smallest coordinates: one length for a segment, and
   one length per axis for a rectangle or a box. Default :math:`0\,\mathrm{m}`
   on every axis. In axisymmetric coordinates the first coordinate is the
   radius, so it is :math:`0\,\mathrm{m}` or more.

``grading``
   The ratio of the length of each cell to the one before it, along each axis:
   one number for a segment, and one per axis for a rectangle or a box.
   Positive numbers. Default 1 on every axis, a uniform mesh. A ratio
   :math:`g` above 1 makes the cells grow from the start of the axis to its
   end, and a ratio below 1 makes them shrink. Along an axis of length
   :math:`\ell` with :math:`n` cells, cell :math:`i` (from 0) has the length
   :math:`\ell\, g^{i} (g - 1)/(g^{n} - 1)`.

.. list-table:: The keys of a generated mesh
   :header-rows: 1
   :widths: 18 22 60

   * - Form
     - Key
     - Value
   * - ``segment``
     - ``length``
     - The length :math:`\ell` of the interval along :math:`x`, in m. A
       positive length. Required.
   * -
     - ``cells``
     - The number of cells. An integer of at least 1. Required.
   * - ``rectangle``
     - ``size``
     - The width and the height, ``[width, height]``. Positive lengths.
       Required.
   * -
     - ``cells``
     - The number of cells along :math:`x` and :math:`y`, ``[columns,
       rows]``. Integers of at least 1. Required.
   * -
     - ``cell_type``
     - ``quadrilateral`` or ``triangle``. Default ``quadrilateral``.
   * - ``box``
     - ``size``
     - The lengths along :math:`x`, :math:`y` and :math:`z`. Positive
       lengths. Required.
   * -
     - ``cells``
     - The number of cells along :math:`x`, :math:`y` and :math:`z`.
       Integers of at least 1. Required.
   * -
     - ``cell_type``
     - ``hexahedron`` or ``tetrahedron``. Default ``hexahedron``.

The boundaries of a generated mesh are ``left`` and ``right`` at the smallest
and the largest :math:`x`, ``bottom`` and ``top`` at the smallest and the
largest :math:`y`, and ``back`` and ``front`` at the smallest and the largest
:math:`z`. A segment has ``left`` and ``right``, a rectangle adds ``bottom``
and ``top``, and a box has all six.

.. code-block:: yaml

   mesh:
     rectangle:
       size: [0.1 m, 0.05 m]
       cells: [10, 5]
       cell_type: triangle

A tube of inner radius :math:`0.1\,\mathrm{m}` in axisymmetric coordinates,
with rows that shrink towards its top:

.. code-block:: yaml

   mesh:
     rectangle: {origin: [0.1 m, 0 m], size: [0.05 m, 0.3 m], cells: [10, 60], grading: [1, 0.98]}
     coordinate_system: axisymmetric

A mesh file takes these keys:

``file``
   The path of the mesh file, in a format that meshio reads, such as a Gmsh
   ``.msh`` file. Required for this form.

``length_unit``
   The unit of the coordinates of the file, a unit of length such as ``m`` or
   ``mm``. Required with ``file``, because a mesh file does not state the unit
   of its coordinates. A generated mesh refuses it, because its sizes carry
   their units.

The physical groups of a Gmsh file, or the element blocks of an Exodus file,
are the regions. Its lower-dimensional groups are the boundaries, named
after their physical names, or ``boundary_<id>`` when they have none. The
command ``dualmesh mesh FILE`` lists the side sets of a mesh file.
:doc:`/user_guide/meshes` describes mesh files.

.. code-block:: yaml

   mesh:
     file: wrench.msh
     length_unit: mm

``coordinate_system``
   ``cartesian`` or ``axisymmetric``. Default ``cartesian``. In axisymmetric
   coordinates a plane mesh describes a body of revolution: its first
   coordinate is the radius :math:`r` and its second the axial coordinate
   :math:`z`. Axisymmetric coordinates need a mesh of space dimension 2, a
   rectangle or a plane mesh file, so a segment and a box refuse them.

.. _input-method:

method
~~~~~~

The section ``method`` names the discretisation of the problem: ``FEM``, the
finite element method, or ``DMCDM``, the dual mesh control domain method.
A physics or a model can also name it with its own key ``method``. A problem
takes one method, so the input refuses two different methods at the second.

Without a method, a problem with a ``fluid_dynamics`` of the ``equal_order``
formulation takes DMCDM, with any solid or heat conduction beside the flow,
and every other problem takes FEM.

.. code-block:: yaml

   method: DMCDM

.. _input-physics:

physics
~~~~~~~

The section ``physics`` holds one entry per physics, each a mapping of its
options. An unknown physics or option is refused with the nearest valid name.
Each physics solves for its fields and reads the properties listed under
:ref:`input-properties`.

.. list-table:: The physics
   :header-rows: 1
   :widths: 22 28 50

   * - Physics
     - Fields
     - Options
   * - ``heat_conduction``
     - ``temperature``
     - ``method``
   * - ``solid_mechanics``
     - ``displacement``
     - ``formulation``, ``kinematics``, ``strain_energy``, ``method``
   * - ``fluid_dynamics``
     - ``velocity``, ``pressure``
     - ``formulation``, ``method``

.. rubric:: heat_conduction

``heat_conduction`` solves the energy balance for the temperature :math:`T`
in K,

.. math::

   \rho c_p \frac{\partial T}{\partial t} + \nabla\cdot\mathbf{q} = s,
   \qquad \mathbf{q} = -k\nabla T,

with :math:`\rho` the density in :math:`\mathrm{kg/m^3}`, :math:`c_p` the
specific heat capacity in :math:`\mathrm{J/(kg\,K)}`, :math:`t` the time in s,
:math:`\mathbf{q}` the heat flux in :math:`\mathrm{W/m^2}`, :math:`k` the
thermal conductivity in :math:`\mathrm{W/(m\,K)}` and :math:`s` the
volumetric heat source in :math:`\mathrm{W/m^3}`. A steady body at rest reads
neither :math:`\rho` nor :math:`c_p`. With ``fluid_dynamics`` the flow carries
the heat, so the problem reads both, steady or transient.

.. code-block:: yaml

   physics:
     heat_conduction: {}

.. rubric:: solid_mechanics

``solid_mechanics`` solves the balance of momentum of an elastic solid for
its displacement :math:`\mathbf{u}` in m.

``formulation``
   ``plane_strain`` or ``plane_stress`` on a plane mesh in Cartesian
   coordinates, where it is required. The mesh sets the formulation in three
   dimensions (``three_dimensional``) and in axisymmetric coordinates
   (``axisymmetric``), so leave the key out there. A formulation that differs
   from the one the mesh sets is refused. A segment mesh is refused.

``kinematics``
   ``small`` or ``finite``. Default ``small``, the linear elasticity of small
   strains and rotations, given by the Young modulus :math:`E` and the
   Poisson ratio :math:`\nu`.

``strain_energy``
   The strain energy per undeformed volume :math:`\psi` in
   :math:`\mathrm{J/m^3}` of a solid at finite strain: ``neo-Hookean``,
   ``Saint_Venant_Kirchhoff`` or a formula. Default ``neo-Hookean``. A solid
   at small strain refuses this key, because every energy has linear
   elasticity with :math:`E` and :math:`\nu` as its small-strain limit.

A formula of ``strain_energy`` reads the invariants of the right
Cauchy-Green tensor :math:`\mathbf{C} = \mathbf{F}_d^T\mathbf{F}_d`, with
:math:`\mathbf{F}_d` the deformation gradient:

- ``I_C``, the trace of :math:`\mathbf{C}`,
- ``trace_C_squared``, the trace of :math:`\mathbf{C}^2`,
- ``J``, the determinant of :math:`\mathbf{F}_d`, the ratio of the deformed
  volume to the undeformed one,
- ``shear_modulus`` :math:`G = E/(2(1+\nu))` and ``first_Lame_parameter``
  :math:`\lambda = E\nu/((1+\nu)(1-2\nu))`, both in Pa.

The two named energies are

.. math::

   \psi_\text{neo-Hookean} = \frac{G}{2}(I_C - 3) - G\ln J
   + \frac{\lambda}{2}(\ln J)^2,

.. math::

   \psi_\text{Saint Venant-Kirchhoff} = \frac{\lambda}{8}(I_C - 3)^2
   + \frac{G}{4}\left(\operatorname{tr}(\mathbf{C}^2) - 2 I_C + 3\right),

with :math:`I_C = \operatorname{tr}\mathbf{C}`. The code differentiates the
energy into the first Piola-Kirchhoff stress
:math:`\mathbf{P} = \partial\psi/\partial\mathbf{F}_d` in Pa, and into the
derivative of :math:`\mathbf{P}` that Newton's method needs.

At finite strain the density, the pressure and the thermal expansion take
the meaning of a body that deforms:

- ``density`` is the density :math:`\rho_0` in :math:`\mathrm{kg/m^3}` per
  undeformed volume. The inertia
  :math:`\rho_0\,\partial^2\mathbf{u}/\partial t^2` and the weight
  :math:`\rho_0\mathbf{g}` act per undeformed volume, with :math:`t` the time
  in s and :math:`\mathbf{g}` the gravitational acceleration in
  :math:`\mathrm{m/s^2}`.
- A ``pressure`` :math:`p` in Pa follows the deformed boundary. Per undeformed
  area its traction is :math:`-p\,J\,\mathbf{F}_d^{-T}\mathbf{N}`, with
  :math:`\mathbf{N}` the outward unit normal of the undeformed boundary.
- A thermal expansion stretches the material by the thermal stretch
  :math:`\vartheta = 1 + \alpha(T - T_0)` in every direction, with
  :math:`\alpha` the ``linear_expansion_coefficient`` in
  :math:`\mathrm{K^{-1}}`, :math:`T` the temperature in K and :math:`T_0` the
  ``stress_free_temperature`` in K. The formula of ``strain_energy`` then
  gives the energy :math:`\psi` per volume of the heated material free of
  stress, as a function of the elastic part :math:`\mathbf{F}_d/\vartheta` of
  the deformation gradient, and the energy per undeformed volume is
  :math:`\vartheta^3\psi`.

A solid at finite strain takes every formulation. In ``plane_stress`` the
code finds, at each point, the stretch of the thickness :math:`\lambda_3`
(dimensionless) at which the stress across the thickness is zero. The
stretch enters :math:`\mathbf{F}_d`, so a ``pressure`` acts on the deformed
thickness. A state without such a stretch is inadmissible: Newton's method
shortens its step there, or the step is cut. With ``Saint_Venant_Kirchhoff``
and no thermal expansion this happens once
:math:`C_{11} + C_{22} \ge 3 + 2G/\lambda`, with :math:`C_{11}` and
:math:`C_{22}` the in-plane diagonal components of :math:`\mathbf{C}`. At
:math:`\nu = 0.3` the limit is an equibiaxial stretch of about 1.47.

.. code-block:: yaml

   physics:
     solid_mechanics:
       formulation: plane_strain
       kinematics: finite
       strain_energy: "shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + first_Lame_parameter/2*log(J)^2"

A solid at small strain is plastic when the properties give a
``yield_stress`` :math:`\sigma_y` in Pa, the von Mises equivalent stress at
the onset of plastic flow. The plastic flow follows the J2 theory: the strain
splits into an elastic and a plastic part, the plastic part grows along the
deviator of the stress and keeps the volume, and the von Mises stress
:math:`\sigma_\text{eq}` stays at most :math:`\sigma_y`. The yield stress is a
constant or a formula of ``equivalent_plastic_strain``, the accumulated
plastic strain :math:`\bar\varepsilon_p` (dimensionless), and of
``temperature`` when the problem has ``heat_conduction``. The formula is the
isotropic hardening of the material. A solid at finite strain refuses
``yield_stress``, because the solid module updates a plastic stress at
finite strain by a hypoelastic rate, which has no multiplicative split of the
deformation gradient.

.. code-block:: yaml

   physics:
     solid_mechanics: {formulation: plane_stress}
   properties:
     Young_modulus: 200 GPa
     Poisson_ratio: 0.3
     yield_stress: "[250 MPa] + [1 GPa]*equivalent_plastic_strain"

In uniaxial tension by :math:`\sigma = 300\,\mathrm{MPa}`, this material
reaches :math:`\bar\varepsilon_p = 0.05`, so its axial strain is
:math:`\sigma/E + \bar\varepsilon_p`. The test
``test_a_bar_beyond_yield_has_the_strain_of_its_hardening`` in
``tests/python/test_input_capabilities.py`` checks it.

.. rubric:: fluid_dynamics

``fluid_dynamics`` solves the balance of mass and momentum of an
incompressible Newtonian fluid for its velocity :math:`\mathbf{v}` in :math:`\mathrm{m/s}` and
its pressure :math:`p` in Pa.

``formulation``
   ``equal_order``, ``Taylor-Hood`` or ``penalty``. ``equal_order`` takes
   the velocity and the pressure of the same order with a pressure
   stabilisation, and runs with both methods and on the linear cells of a
   generated mesh. ``Taylor-Hood`` takes a quadratic velocity and a linear
   pressure. It needs ``method: FEM`` and a mesh file of quadratic cells.
   ``penalty`` eliminates the pressure by a penalty on the divergence of the
   velocity. Default ``Taylor-Hood`` with ``method: FEM`` on a mesh of
   quadratic cells, since it needs no stabilisation, and ``equal_order``
   otherwise.

.. code-block:: yaml

   physics:
     fluid_dynamics:
       formulation: equal_order

.. rubric:: Couplings

The input couples its physics from the physics and the properties it holds:

- ``heat_conduction`` with ``solid_mechanics`` and the properties
  ``linear_expansion_coefficient`` and ``stress_free_temperature`` gives a
  thermal expansion of the solid. The solve takes the two properties
  together, and refuses them without ``heat_conduction``. The thermal strain
  is :math:`\alpha(T)\,(T - T_0)` in every direction, with :math:`\alpha` the
  ``linear_expansion_coefficient`` in :math:`\mathrm{K^{-1}}`, :math:`T` the
  temperature in K and :math:`T_0` the ``stress_free_temperature`` in K.
  :math:`\alpha(T)` is the secant (mean) coefficient between :math:`T_0` and
  :math:`T`, so a formula such as
  ``"[1.0e-5 1/K] + [4.0e-9 1/K^2]*temperature"`` gives the strain of a
  coefficient that grows with the temperature.
- ``heat_conduction`` with ``fluid_dynamics`` gives a heated flow, which
  carries the heat.
- A heated flow under gravity takes the Boussinesq buoyancy
  :math:`-\rho\beta(T - T_0)\mathbf{g}` on top of its weight
  :math:`\rho\mathbf{g}`, with :math:`\beta` the volumetric expansion
  coefficient in :math:`\mathrm{K^{-1}}`, :math:`T_0` the reference temperature in K and
  :math:`\mathbf{g}` the gravitational acceleration in
  :math:`\mathrm{m/s^2}`.

.. _input-properties:

properties
~~~~~~~~~~

The section ``properties`` maps each property by its standard name to its
value. The value is a constant, a formula, a table or a mapping of regions to
values (see :ref:`input-values`). A property takes its unit from the
vocabulary, so a ``unit`` key on a standard name is refused.

.. list-table:: The properties that the physics read
   :header-rows: 1
   :widths: 30 14 30 26

   * - Property
     - Unit
     - Read by
     - Value
   * - ``thermal_conductivity``
     - ``W/(m*K)``
     - ``heat_conduction``, always
     - scalar, per region
   * - ``density``
     - ``kg/m^3``
     - ``heat_conduction`` with time terms or with a flow,
       ``solid_mechanics`` with time terms or under gravity,
       ``fluid_dynamics`` always
     - per region for ``heat_conduction``, one constant for
       ``solid_mechanics`` and ``fluid_dynamics``
   * - ``specific_heat_capacity``
     - ``J/(kg*K)``
     - ``heat_conduction`` with time terms or with a flow
     - per region
   * - ``volumetric_heat_source``
     - ``W/m^3``
     - ``heat_conduction``, when given
     - per region
   * - ``Young_modulus``
     - ``Pa``
     - ``solid_mechanics``, always
     - per region at small strain, one constant at finite strain
   * - ``Poisson_ratio``
     - ``1``
     - ``solid_mechanics``, always
     - per region at small strain, one constant at finite strain
   * - ``linear_expansion_coefficient``
     - ``1/K``
     - ``solid_mechanics`` with ``heat_conduction``, when given
     - one constant or one formula of ``temperature``, the other fields
       and the coordinates
   * - ``stress_free_temperature``
     - ``K``
     - ``solid_mechanics`` with ``heat_conduction``, when given
     - one constant
   * - ``dynamic_viscosity``
     - ``Pa*s``
     - ``fluid_dynamics``, always
     - one constant
   * - ``volumetric_expansion_coefficient``
     - ``1/K``
     - ``fluid_dynamics`` with ``heat_conduction`` under gravity
     - one constant
   * - ``reference_temperature``
     - ``K``
     - ``fluid_dynamics`` with ``heat_conduction`` under gravity
     - one constant
   * - ``yield_stress``
     - ``Pa``
     - ``solid_mechanics`` at small strain, when given
     - one constant or one formula of ``equivalent_plastic_strain`` and
       ``temperature``

A problem has time terms in a transient run and in an eigenvalue study.
"Per region" means a constant, a formula or a table, in every region or one per
region.
"One constant" means the same constant in every region. The heat source can
read the fields, as the source :math:`-\gamma k (T - T_a)` of the cooling fin
of ``examples/cooling_fin.py`` does, with
:math:`\gamma = 400\,\mathrm{m^{-2}}` the ratio of the convection over the
perimeter of the fin to the conduction along it, and
:math:`T_a = 293\,\mathrm{K}` the ambient temperature:

.. code-block:: yaml

   properties:
     thermal_conductivity: 200 W/(m*K)
     volumetric_heat_source: "-[400 1/m^2]*[200 W/(m*K)]*(temperature - [293 K])"

The solve checks the properties against the physics:

- A property that a physics needs and the input leaves out is asked for by
  name, such as ``heat_conduction needs the property thermal_conductivity``.
- A property that no physics of the problem reads is refused, such as a
  ``density`` in a steady ``heat_conduction``.
- ``body_force_density`` is refused. Give ``gravity`` for the weight of a
  body.
- A property takes the whole quantity, so a component name such as
  ``thermal_conductivity_xx`` is refused.
- ``thermal_conductivity`` takes a scalar.
- The names ``x``, ``y``, ``z``, ``r``, ``time``, ``time_step`` and
  ``cell_size`` are refused, because a formula reads them as the position,
  the time and the cell size. A physical constant is refused, because its
  value is fixed.

.. code-block:: yaml

   properties:
     Young_modulus: 200 GPa
     Poisson_ratio: 0.3
     density: 7850 kg/m^3

.. _input-gravity:

gravity
~~~~~~~

The section ``gravity`` is the gravitational acceleration :math:`\mathbf{g}`
in :math:`\mathrm{m/s^2}`, a list of one quantity per direction of space. A
solid and a flow take its force :math:`\rho\mathbf{g}` per volume, so a
problem with gravity needs ``density``. A solid at finite strain takes the
force per undeformed volume, which stays the same as the body deforms.

The solve refuses gravity on a problem without ``solid_mechanics`` or
``fluid_dynamics``, and a list whose length differs from the dimension of the
mesh. In axisymmetric coordinates gravity acts along the axis :math:`z`, so
its radial component is 0.

.. code-block:: yaml

   gravity: [0 m/s^2, -9.81 m/s^2]

.. _input-models:

models
~~~~~~

The section ``models`` holds balance laws of your own, one entry per law. A
problem takes its equations from ``physics``, from ``models`` or from both.

Each entry states the balance law

.. math::

   \frac{\partial m}{\partial t} + \nabla\cdot\mathbf{F} = s

for one scalar unknown :math:`u`, at each point of the domain. Here :math:`t`
is the time in s, :math:`m` the storage, the amount of the balanced quantity
per volume, :math:`\mathbf{F}` its flux and :math:`s` its source per volume.
:math:`\mathbf{F}` takes the unit of :math:`m` times m/s, and :math:`s` the
unit of :math:`m` per s, so the three terms agree. Each internal state :math:`q` follows its own
rate law

.. math::

   \frac{\mathrm{d}q}{\mathrm{d}t} = g

at each point, with :math:`g` its rate. A state has no flux: it changes only
where it is. The terms and the rates are formulas of the unknown, its gradient
``grad(...)``, the states, the fields of the other physics and models, the
properties, the position and the time.

The model of a hydrogen concentration :math:`c` in :math:`\mathrm{mol/m^3}` that
diffuses and fills traps of density :math:`N` in :math:`\mathrm{mol/m^3}`,
with :math:`\theta` the dimensionless occupancy of the traps:

.. code-block:: yaml

   models:
     hydrogen_trapping:
       unknown: {hydrogen_concentration: {unit: mol/m^3}}
       states:
         trap_occupancy:
           unit: "1"
           initial: 0
           rate: "trapping_rate*hydrogen_concentration*(1 - trap_occupancy) - release_rate*trap_occupancy"
       storage: "hydrogen_concentration + trap_density*trap_occupancy"
       flux: "-diffusivity*grad(hydrogen_concentration)"
       flux_name: hydrogen_flux

   properties:
     trap_density: {unit: mol/m^3, value: 10 mol/m^3}
     trapping_rate: {unit: m^3/(mol*s), value: "[1e3 m^3/(mol*s)]*exp(-[5.7 kJ/mol]/(molar_gas_constant*temperature))"}
     release_rate: {unit: 1/s, value: "[1e7 1/s]*exp(-[60 kJ/mol]/(molar_gas_constant*temperature))"}
     diffusivity: {unit: m^2/s, value: "[1e-7 m^2/s]*exp(-[5.7 kJ/mol]/(molar_gas_constant*temperature))"}

   initial_conditions:
     hydrogen_concentration: 0 mol/m^3

   boundary_conditions:
     left: {hydrogen_concentration: 1 mol/m^3}
     right: {hydrogen_flux: 0 mol/(m^2*s)}

The storage is :math:`c + N\theta`, so the traps hold part of the hydrogen.
The occupancy rises at the rate :math:`k c (1 - \theta)` and falls at the rate
:math:`p\theta`, with the trapping rate :math:`k` in
:math:`\mathrm{m^3/(mol\,s)}` and the release rate :math:`p` in
:math:`\mathrm{1/s}`. The properties read the ``temperature``, so the problem
also needs ``heat_conduction`` or another model with the unknown
``temperature``.

The keys of an entry are:

``unknown``
   The field that the law solves for: a standard name, such as
   ``temperature``, or a mapping of one name to its ``unit``. Required. A name
   outside the vocabulary needs its unit.

``states``
   The internal states, each a mapping of its ``unit``, its ``initial`` value
   at every point and its ``rate``, the formula of :math:`g`. ``rate`` is
   required, and ``initial`` is 0 when you leave it out.

``storage``, ``flux``, ``source``
   The formulas of :math:`m`, :math:`\mathbf{F}` and :math:`s`. A term you
   leave out is zero.

``expressions``
   Named formulas, each of the form ``{unit: ..., value: ...}``, that the terms,
   the rates and the later expressions read by name. A postprocessor reads an
   expression too.

``flux_name``
   The name of :math:`\mathbf{F}`, which ``boundary_conditions`` and a
   ``boundary_flux`` postprocessor take. A name outside the vocabulary takes
   the unit of :math:`\mathbf{F}`.

``method``
   ``FEM`` or ``DMCDM``. One problem takes one method.

Write the formulas of a model as the :ref:`formulas <input-formulas>` of a
property. A model reads a property by its name, so the property section holds
the numbers of the model, as it holds those of a physics. The check refuses a
model whose terms disagree in unit, whose rate has a unit other than that of
its state per s, or whose formula names a quantity that the problem does not
hold. The message names the key, as in ``models.hydrogen_trapping: ... the
terms disagree in dimension``.

In an :ref:`eigenvalue study <input-eigenvalues>`, the source of a model may
read the symbol ``eigenvalue``, the eigenvalue :math:`\lambda` of the study.
Its unit follows the problem. A problem with a time derivative, the time terms
of a physics or the storage of a model, has :math:`\lambda` in
:math:`\mathrm{1/s}`. A problem without one has a dimensionless
:math:`\lambda`, which the source alone defines. The fast group of a reactor
writes its fission source this way, with :math:`\lambda = 1/k` and the
effective multiplication factor :math:`k`:

.. code-block:: yaml

   models:
     fast_group:
       unknown: {fast_flux: {unit: 1/(m^2*s)}}
       flux: "-fast_diffusion_coefficient*grad(fast_flux)"
       source: "-(fast_absorption + fast_to_thermal_scattering)*fast_flux + eigenvalue*thermal_fission_production*thermal_flux"

   eigenvalues: {count: 1}

The fluxes :math:`\phi_1` of ``fast_flux`` and :math:`\phi_2` of
``thermal_flux`` are in :math:`\mathrm{1/(m^2\,s)}`, and the cross sections
in :math:`\mathrm{1/m}`, so the source holds no factor of one second.
:doc:`/tutorials/criticality` solves the whole core.

The other sections take the names of a model:

- ``initial_conditions`` takes the unknown. A transient model with a storage
  needs it. A state takes its initial value in its ``initial`` key.
- ``boundary_conditions`` takes the unknown, whose value it holds on the
  boundary, and the ``flux_name``, whose value is the outward normal flux
  :math:`\mathbf{F}\cdot\mathbf{n}`, with :math:`\mathbf{n}` the outward unit
  normal. A positive value leaves the body. Both take a constant or a
  formula, such as a flux that grows with the concentration at the surface. A
  boundary without a condition on the model has no flux through it.
- ``postprocessors`` take the unknown, the states and the expressions, and
  ``boundary_flux`` takes the ``flux_name``. A state and an expression live
  at the integration points, so they take ``integral``, ``average``,
  ``maximum`` and ``minimum``, each over the averages of the points of each
  cell.
- ``output`` writes the unknown, and ``mesh_adaptivity`` reads it.

In Python, ``model(name, **entry)`` adds an entry. Give the model before the
properties whose formulas read its unknown or its states.

The solve has these limits:

- The unknown and the states are scalars, and a model holds no ``inertia``.
- Only the source reads ``eigenvalue``, and only in a problem with the
  ``eigenvalues`` section, since every other run has no eigenvalue.
- A model reads constants and formulas, so a property that it reads takes no
  table.
- A flux through a boundary goes in ``boundary_conditions`` under the
  ``flux_name``, so the solve refuses the key ``boundary_fluxes``.
- A steady problem holds each state at the root of its rate, where the state
  stops changing, so it refuses a rate that reads no state.
- A transient problem with states takes no ``mesh_adaptivity``, since a new
  mesh does not keep the history of the integration points.
- The step control of a transient run measures the change of the unknowns,
  and not that of the states.

:ref:`theory-user-models` describes how the solve discretises a model.

.. _input-initial-conditions:

initial_conditions
~~~~~~~~~~~~~~~~~~

The section ``initial_conditions`` gives the value of each field at the start
of a transient run, by its standard name. A scalar field takes one constant or
one formula of the coordinates. A vector field takes a list of them, one per
direction. A formula reads the position, as ``x``, ``y`` and ``z``, or ``r``
and ``z`` in axisymmetric coordinates, and constants. It refuses the fields,
the properties and the time. A table is refused.

- A transient ``heat_conduction`` needs the initial ``temperature``.
- An initial ``displacement`` needs the ``time`` section, because it starts a
  run with inertia.
- A flow takes an initial ``velocity`` and ``pressure``.
- A field that no physics or model of the problem solves for is refused, and so is a
  component name such as ``displacement_x``.

- An eigenvalue study refuses the section, because it finds the modes of the
  problem from no starting state.

.. code-block:: yaml

   initial_conditions:
     temperature: 573 K

A slab whose faces are at :math:`300\,\mathrm{K}` and
:math:`305\,\mathrm{K}`, :math:`0.1\,\mathrm{m}` apart, starts at its steady
temperature, which rises by :math:`50\,\mathrm{K/m}` along :math:`x`:

.. code-block:: yaml

   initial_conditions:
     temperature: "[300 K] + [50 K/m]*x"

A solid at finite strain that starts rotated by :math:`0.5\,\mathrm{rad}` as
a rigid body, at rest and free of load, stays free of stress. The test
``test_a_rigid_rotation_leaves_a_solid_at_finite_strain_free_of_stress`` in
``tests/python/test_input_capabilities.py`` gives it this start:

.. code-block:: yaml

   initial_conditions:
     displacement: ["(cos(0.5) - 1)*x - sin(0.5)*y", "sin(0.5)*x + (cos(0.5) - 1)*y"]

.. _input-boundary-conditions:

boundary_conditions
~~~~~~~~~~~~~~~~~~~

The section ``boundary_conditions`` maps each boundary to the conditions on
it, or to the word ``symmetry``. Each key sets a field or a flux through the
boundary. A boundary of a
generated mesh that does not exist is refused with the nearest valid name. A
boundary that the input leaves out, or a physics without a condition on it,
takes the natural condition: no heat flux for ``heat_conduction``, and no
traction for ``solid_mechanics`` and ``fluid_dynamics``.

.. list-table:: The keys of a boundary
   :header-rows: 1
   :widths: 26 22 52

   * - Key
     - Physics
     - Value
   * - ``temperature``
     - ``heat_conduction``
     - The temperature in K.
   * - ``heat_flux``
     - ``heat_conduction``
     - The outward heat flux :math:`\mathbf{q}\cdot\mathbf{n}` in
       :math:`\mathrm{W/m^2}`, positive when heat leaves the body.
   * - ``convection``
     - ``heat_conduction``
     - A mapping of ``heat_transfer_coefficient`` and
       ``ambient_temperature``. The outward heat flux is
       :math:`h_c(T - T_a)`.
   * - ``radiation``
     - ``heat_conduction``
     - A mapping of ``emissivity`` and ``ambient_temperature``. The outward
       heat flux is :math:`\sigma_\mathrm{SB}\,\epsilon\,(T^4 - T_a^4)`.
   * - ``displacement``
     - ``solid_mechanics``
     - A list of one value per direction, in m.
   * - ``displacement_x``, ``displacement_y``, ``displacement_z``
     - ``solid_mechanics``
     - One component in m. The other components stay free.
   * - ``traction``
     - ``solid_mechanics``
     - The force per area, a list of one value per direction, in Pa. At
       finite strain it is :math:`\mathbf{P}\mathbf{N}` per undeformed area,
       with :math:`\mathbf{N}` the outward unit normal of the undeformed
       boundary.
   * - ``pressure``
     - ``solid_mechanics`` in a problem without a flow
     - The pressure :math:`p` in Pa, the load :math:`-p\,\mathbf{n}`. At
       finite strain it acts on the deformed boundary.
   * - ``velocity``
     - ``fluid_dynamics``
     - A list of one value per direction, in :math:`\mathrm{m/s}`.
   * - ``velocity_x``, ``velocity_y``, ``velocity_z``
     - ``fluid_dynamics``
     - One component in :math:`\mathrm{m/s}`.
   * - ``pressure``
     - ``fluid_dynamics``
     - The pressure in Pa.

In the table, :math:`h_c` is the heat transfer coefficient in
:math:`\mathrm{W/(m^2\,K)}`, :math:`\epsilon` the dimensionless emissivity
and :math:`\sigma_\mathrm{SB} = 5.670374419 \times 10^{-8}\,\mathrm{W/(m^2\,K^4)}`
the Stefan-Boltzmann constant, the physical constant ``Stefan_Boltzmann_constant``.
Both conditions prescribe the heat flux of the row through the one route of a
prescribed flux. In axisymmetric coordinates the components are
``displacement_r``, ``displacement_z``, ``velocity_r`` and ``velocity_z``.

A scalar value, each entry of a list, and each constant of ``convection`` and
``radiation``, is a constant or a formula. The input refuses a table and a
mapping of regions as a boundary value, since a condition holds one constant
or formula along its boundary. The traction of a remote stress
:math:`\sigma_\infty` along :math:`x` on an arc of radius :math:`r` reads:

.. code-block:: yaml

   boundary_conditions:
     outer:
       traction: ["[100 MPa]*x/sqrt(x^2 + y^2)", "0 Pa"]

The solve refuses a key whose physics the problem does not have, such as a
``heat_flux`` without ``heat_conduction`` or a ``traction`` without
``solid_mechanics``. ``convection`` and ``radiation`` refuse a missing or an
unknown constant. A list of the wrong length is refused.

The boundary conditions of ``examples/bus_bar.py`` read:

.. code-block:: yaml

   boundary_conditions:
     left:
       temperature: 40 degC
     right:
       temperature: 10 degC
     top:
       convection:
         heat_transfer_coefficient: 75 W/(m^2*K)
         ambient_temperature: 0 degC

A boundary takes several keys, one per physics, as a heated and clamped edge
does:

.. code-block:: yaml

   boundary_conditions:
     left:
       temperature: 400 K
       displacement_x: 0 m

The word ``symmetry`` takes the place of the mapping of a boundary, and makes
the boundary a plane of symmetry. Each physics of the problem applies its own
condition there:

- ``solid_mechanics``: the displacement normal to the boundary is zero, and
  the tangential traction is zero.
- ``fluid_dynamics``: the velocity normal to the boundary is zero, and so is
  the tangential traction.
- ``heat_conduction`` and a model: the flux through the boundary is zero.

A plane of symmetry is normal to a coordinate axis, so the coordinate across
it is the same at all of its nodes. The input refuses another key on a
boundary of symmetry.

.. code-block:: yaml

   boundary_conditions:
     left: symmetry
     bottom: symmetry

.. _input-time:

time
~~~~

The section ``time`` makes the run transient. Without it the run is steady.
The run starts at the time 0 s.

``end_time``
   The time at which the run ends, a positive time. Required.

``initial_time_step``
   The first time step :math:`\Delta t`, a positive time. Required.

``step_control``
   How the run chooses each step after the first. A mapping whose key ``by``
   names the controller, with the optional setting of that controller.
   Without it, the step follows the estimated error of each step with the
   relative tolerance :math:`10^{-3}`, as ``{by: error}`` does.

.. list-table:: The step controllers
   :header-rows: 1
   :widths: 16 34 50

   * - ``by``
     - Setting and default
     - Step
   * - ``error``
     - ``relative_tolerance``, a number above 0 and at most 1. Default
       :math:`10^{-3}`.
     - The step that keeps the estimated local error of each variable,
       against its scale (:doc:`solving`), at the tolerance. A step whose
       error exceeds it is repeated with a smaller step.
   * - ``iterations``
     - ``target_iterations``, an integer of at least 1. Default 4.
     - A larger step after a step that took fewer nonlinear iterations than
       the target, and a smaller step after one that took more.
   * - ``fixed``
     - no setting
     - The initial time step throughout, with the last step shortened so
       that the run ends at ``end_time``.

The defaults are those of :meth:`~dualmesh.Problem.solve_transient`, which the
input leaves to it. A setting of another controller is refused, such as
``relative_tolerance`` with ``by: fixed``.

The time scheme is backward Euler, and a solid with inertia takes the
generalised-:math:`\alpha` method [ChungHulbert1993]_.
:doc:`/user_guide/solving` describes the transient solve.

.. code-block:: yaml

   time:
     end_time: 40 s
     initial_time_step: 0.1 s
     step_control: {by: fixed}

.. _input-eigenvalues:

eigenvalues
~~~~~~~~~~~

The section ``eigenvalues`` makes the problem an eigenvalue study, in place of
a steady or a transient run. Every field varies in time as
:math:`\hat{u}\, e^{-\lambda t}`, with :math:`\hat{u}` the mode, :math:`t`
the time in s and :math:`\lambda` the eigenvalue in :math:`\mathrm{1/s}`.
The time terms of the problem then define :math:`\lambda`: a storage
:math:`c\,\partial u/\partial t` becomes :math:`-\lambda c\,\hat{u}`, and an
inertia :math:`\rho\,\partial^2 u/\partial t^2` becomes
:math:`\lambda^2 \rho\,\hat{u}`. The source of a model may also read
:math:`\lambda` as the symbol ``eigenvalue`` (see :ref:`input-models`). A
problem without a time derivative takes :math:`\lambda` from that source
alone, and :math:`\lambda` is then dimensionless. The prescribed values of the
boundary conditions are zero in a mode. Both keys are optional.

``count``
   The number of eigenvalues, an integer of at least 1. Default 1.

``near``
   The value that the eigenvalues are closest to, in the unit of
   :math:`\lambda`, or the angular frequency :math:`\omega` in
   :math:`\mathrm{rad/s}` that the modes of a vibration are closest to.
   Default 0, so the study finds the slowest modes, or the smallest
   :math:`|\lambda|`.

The eigenvalues of a cooling body, whose time term is the storage of heat,
are its rates of decay. A vibration is a problem whose only time term is an
inertia. Its eigenvalues are :math:`\lambda = i\omega`, with :math:`\omega`
the angular frequency in :math:`\mathrm{rad/s}`, and the study also gives its
natural frequencies :math:`f = |\operatorname{Im}\lambda|/(2\pi)` in Hz.

A problem with ``eigenvalues`` refuses the sections ``time``,
``mesh_adaptivity``, ``initial_conditions`` and ``postprocessors``, since a
mode has no scale of its own. The results hold the eigenvalues, closest to
``near`` first, and the fields of the first mode. The output writes each mode,
with its largest value 1, as ``<file>_mode_<number>.vtu``.

.. code-block:: yaml

   eigenvalues: {count: 6, near: 0 1/s}

A bar of steel of length :math:`L = 1\,\mathrm{m}`, fixed at one end and free
at the other, vibrates along its axis at :math:`f_n = (2n - 1)\,c/(4L)` for
:math:`n = 1, 2, \ldots`, with the wave speed :math:`c = \sqrt{E/\rho}` in
:math:`\mathrm{m/s}`. The test
``test_the_natural_frequencies_of_a_bar_are_those_of_its_axial_waves`` in
``tests/python/test_input_capabilities.py`` checks the first two to
:math:`10^{-3}`.

A bare reactor core has no time derivative in its two-group diffusion
equations, so its eigenvalue :math:`\lambda = 1/k` is a number, and the
smallest :math:`\lambda` gives the largest multiplication factor :math:`k`.
:doc:`/tutorials/criticality` computes it.

.. _input-mesh-adaptivity:

mesh_adaptivity
~~~~~~~~~~~~~~~

The section ``mesh_adaptivity`` refines the mesh where the error indicator of
a field is large, and coarsens it where the indicator is small. The indicator
is that of Zienkiewicz and Zhu [ZienkiewiczZhu1987]_, from the difference
between the gradient of the field and its recovered gradient on each cell. For
a vector field the squared indicators of its components add up. Every key is
required, and ``interval`` is read in a transient run only.

``field``
   The field whose indicator drives the adaptivity: a field that a physics
   of the problem solves for, such as ``temperature``.

``refine_fraction``
   A number above 0 and at most 1. The cells refined are the smallest set
   whose squared indicators add up to this fraction of their total, the
   largest first [Dorfler1996]_.

``coarsen_fraction``
   A number from 0 to 1. The cells coarsened are the largest set of refined
   cells, outside those marked for refining, whose squared indicators add up
   to at most this fraction of the total, the smallest first.

``maximum_refinement_level``
   The number of times a cell of the starting mesh can be refined. An
   integer of at least 0.

``interval``
   The number of accepted time steps between two adaptations of a transient
   run. An integer of at least 1. A steady run adapts after each solve, so
   the input refuses an interval there.

A steady run solves, marks and adapts in cycles through
:func:`~dualmesh.solve_with_adaptive_refinement`. Each solve starts from the
solution of the last mesh. The cycles stop when no marked cell can be refined
further, or after :math:`4 d\,n_\mathrm{max} + 1` cycles, with :math:`d` the
dimension of the mesh and :math:`n_\mathrm{max}` the maximum refinement
level. A transient run adapts the mesh through
:func:`~dualmesh.solve_transient_with_adaptive_refinement` after every
``interval`` accepted steps, and goes on from the time reached, with the
solution carried to the new mesh and the step that the controller chose. :doc:`/user_guide/adaptivity`
describes adaptive refinement.

.. code-block:: yaml

   mesh_adaptivity:
     field: temperature
     refine_fraction: 0.5
     coarsen_fraction: 0.05
     maximum_refinement_level: 3

The test ``tests/python/test_parallel.py`` runs this adaptivity on a plate
whose boundary temperature jumps at a corner.

.. _input-postprocessors:

postprocessors
~~~~~~~~~~~~~~

The section ``postprocessors`` maps each name you choose to one reduction of a
field. The reduction is exactly one of these keys:

.. list-table:: The reductions
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Other keys
     - Value and unit
   * - ``integral``
     - none
     - The integral of the field over the domain, in the unit of the field
       times :math:`\mathrm{m^3}` in three dimensions and in axisymmetric
       coordinates, :math:`\mathrm{m^2}` in a plane problem and
       :math:`\mathrm{m}` on a segment.
   * - ``average``
     - none
     - The average of the field over the domain.
   * - ``maximum``
     - none
     - The largest value of the field.
   * - ``minimum``
     - none
     - The smallest value of the field.
   * - ``point_value``
     - ``point``, a list of one length per direction. Required.
     - The value of the field at the point.
   * - ``boundary_flux``
     - ``boundary``, a boundary name. Required.
     - The flow of ``heat_flux`` out through the boundary, in W in three
       dimensions and in axisymmetric coordinates, :math:`\mathrm{W/m}` in a
       plane problem and :math:`\mathrm{W/m^2}` on a segment. Or the
       integral of ``traction`` over the boundary, the force on the body
       through it, in N in three dimensions and in axisymmetric coordinates
       and :math:`\mathrm{N/m}` in a plane problem.
   * - ``line``
     - ``from`` and ``to``, two points, each a list of one length per
       direction, and ``points``, an integer of at least 2. All required.
     - The values of the field at ``points`` evenly spaced points from
       ``from`` to ``to``, both ends included, in the unit of the field.

In axisymmetric coordinates an integral spans the body of revolution. The
field of a reduction is a
scalar field, such as ``temperature`` or ``pressure``, or a component, such as
``displacement_x``. A ``boundary_flux`` of ``heat_flux`` is positive when the
heat leaves the body.

The reductions have these rules:

- ``boundary_flux`` takes ``heat_flux`` in a problem with
  ``heat_conduction``, or the ``flux_name`` of a model, in the unit of its
  flux times the measure of the boundary.
- ``boundary_flux`` takes ``traction``, or one of its components
  ``traction_x``, ``traction_y`` and ``traction_z`` (``traction_r`` and
  ``traction_z`` in axisymmetric coordinates), in a problem with
  ``solid_mechanics``. It is the integral of the traction
  :math:`\boldsymbol{\sigma}\mathbf{n}` over the boundary, with
  :math:`\boldsymbol{\sigma}` the Cauchy stress in Pa and :math:`\mathbf{n}`
  the outward unit normal: the force of the surroundings on the body. On a
  boundary that prescribes the displacement it is the reaction of the
  support, the residual of the equations at the nodes of the boundary, which
  balances the loads to the tolerance of the solve. On a loaded boundary it is
  the load. An axisymmetric integral spans the surface of revolution, and a
  plane one is per unit depth. ``traction`` alone gives one postprocessor per
  component, named ``<name>_x``, ``<name>_y`` and ``<name>_z``.
- A stress is a ``point_value`` of one component, such as ``stress_yy``, or of
  ``von_Mises_stress``. It is the Cauchy stress in Pa that the solid computes
  in the cell that holds the point, at small or finite strain, with the
  thermal strain of a thermal expansion. Its value at a point outside the mesh
  is NaN. The von Mises stress is
  :math:`\sigma_\mathrm{v} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}`,
  with :math:`\mathbf{s}` the deviator of :math:`\boldsymbol{\sigma}`, and it
  holds the out-of-plane stress of a plane strain problem.
- A reduction covers the whole domain, so the input refuses a ``region`` key.
- A ``line`` reads a field, a component, a stress component or the von
  Mises stress, as a ``point_value`` does. It writes ``<file>_<name>.csv``, with ``<file>`` the
  base name of ``output``, so it needs the ``output`` section. The columns
  are ``distance [m]``, the distance along the line from ``from``, then the
  coordinates, such as ``x [m]`` and ``y [m]``, and the field with its unit,
  as in ``velocity_x [m/s]``. A steady run writes the line once, at its end.
  A transient run writes it at each output time, with the column ``time [s]``
  first and the rows of every output time so far. The results give the same
  table with ``solution.line(name)``.

.. code-block:: yaml

   postprocessors:
     heat_through_left: {boundary_flux: heat_flux, boundary: left}
     temperature_bottom: {point_value: temperature, point: [0.05 m, 0 m]}
     hottest: {maximum: temperature}
     reaction: {boundary_flux: traction_y, boundary: jaws}
     edge: {point_value: von_Mises_stress, point: [60 mm, 6 mm, 0 mm]}
     centreline:
       line: velocity_x
       from: [0.5 m, 0 m]
       to: [0.5 m, 1 m]
       points: 101

.. _input-output:

output
~~~~~~

The section ``output`` writes the fields and the postprocessors to files. The
keys ``file`` and ``fields`` are required, and ``interval`` belongs to a
transient run.

``file``
   The base name of the files, relative to the folder of the case file, or
   to the working directory in Python.

``interval``
   A transient run writes the fields at its start, every ``interval``-th
   accepted time step and at its end. An integer of at least 1, and 1 when it
   is left out. A steady run writes the fields once, at its end, and refuses an
   ``interval``.

``fields``
   The list of fields to write: ``temperature``, ``displacement``,
   ``velocity``, ``pressure`` and the unknowns of the models, those of the
   problem.

The section lowers onto the :class:`~dualmesh.Output` group of the solve, with
``interval`` as its ``step_interval``, so the input and Python write the same
files:

- ``<file>.vtu`` in a steady run, and ``<file>_00000.vtu``,
  ``<file>_00001.vtu`` and so on in a transient run, one VTK (Visualization
  Toolkit) unstructured grid file per output time. A run on several processes
  writes one piece per process and a parallel VTU (PVTU) index of the pieces
  in their place.
- ``<file>.pvd`` in a transient run, a collection of those files, which
  ParaView opens as one time series. The files of a transient run that adapts
  its mesh go on in one numbering and one collection.
- ``<file>.csv``, the postprocessors, one column per postprocessor after the
  column ``time [s]``, with the unit in each header, as in ``peak [K]``. The
  run writes it when the input has postprocessors other than lines.
- ``<file>_<name>.csv`` for each ``line`` postprocessor.
- ``<file>_mode_<number>.vtu`` for each mode of an eigenvalue study.

The first process writes the collection and the table.

.. code-block:: yaml

   output:
     file: bus_bar
     fields: [temperature]

.. _input-solver:

solver
~~~~~~

The section ``solver`` sets the nonlinear solve. Both keys are optional.

``relative_tolerance``
   The tolerance of the nonlinear solve, a positive number. Each equation
   converges when its residual is at most this fraction of the size of its
   own terms, so the test holds in any unit. Default :math:`10^{-10}`.

``PETSc_options``
   A mapping of options of PETSc (the Portable, Extensible Toolkit for
   Scientific Computation), without their leading dash, to texts or numbers.
   ``true`` gives the option as a flag without a value, and ``false`` gives
   it the value ``false``, as in Python. Options that start with ``snes_`` go
   to the nonlinear solver of PETSc, and the others replace the choice of the
   linear solver. A leading dash is refused.

The solve refuses ``PETSc_options.snes_rtol``, ``snes_atol`` and
``snes_stol``, which the convergence test does not read. Give
``relative_tolerance`` instead. :doc:`/user_guide/solving` describes the options and
the choice of the linear solver.

.. code-block:: yaml

   solver:
     relative_tolerance: 1e-8
     PETSc_options:
       ksp_type: preonly
       pc_type: lu

.. _input-example:

A whole input
-------------

The bus bar of Example 5.4.3 of [Reddy2024]_ is a plate of
:math:`0.1\,\mathrm{m}` by :math:`0.05\,\mathrm{m}` with a conductivity of
:math:`20\,\mathrm{W/(m\,K)}` and a heat source of
:math:`1\,\mathrm{MW/m^3}`. Its left and right sides are held at
:math:`40` and :math:`10\,^\circ\mathrm{C}`, its bottom is insulated, and its
top loses heat by convection with
:math:`h_c = 75\,\mathrm{W/(m^2\,K)}` to air at :math:`0\,^\circ\mathrm{C}`.
The file ``examples/bus_bar.py`` builds it in Python:

.. literalinclude:: ../../examples/bus_bar.py
   :language: python
   :pyobject: bus_bar

The same input as a case file reads:

.. code-block:: yaml

   mesh:
     rectangle:
       size: [0.1 m, 0.05 m]
       cells: [10, 5]
   method: DMCDM
   physics:
     heat_conduction: {}
   properties:
     thermal_conductivity: 20 W/(m*K)
     volumetric_heat_source: 1 MW/m^3
   boundary_conditions:
     left:
       temperature: 40 degC
     right:
       temperature: 10 degC
     top:
       convection:
         heat_transfer_coefficient: 75 W/(m^2*K)
         ambient_temperature: 0 degC
   postprocessors:
     heat_through_left: {boundary_flux: heat_flux, boundary: left}
     heat_through_right: {boundary_flux: heat_flux, boundary: right}
     temperature_bottom: {point_value: temperature, point: [0.05 m, 0 m]}
     temperature_top: {point_value: temperature, point: [0.05 m, 0.05 m]}
     hottest: {maximum: temperature}
   output:
     file: bus_bar
     fields: [temperature]

The test ``tests/python/test_front_end.py`` checks that the temperature at the
middle of the bottom edge is :math:`83.142\,^\circ\mathrm{C}`, the value of
the book to three decimals.
