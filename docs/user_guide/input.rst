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
     - no method
   * - ``initial_conditions``
     - ``initial_conditions(**values)``
   * - ``boundary_conditions``
     - ``boundary_conditions(boundary, **values)``, once per boundary
   * - ``time``
     - ``time(end_time=..., initial_time_step=..., step_control={...})``
   * - ``mesh_adaptivity``
     - ``mesh_adaptivity(**values)``
   * - ``postprocessors``
     - ``postprocessor(name, **definition)``, once per postprocessor
   * - ``output``
     - ``output(file, interval, fields)``
   * - ``solver``
     - ``solver(**values)``

A second call of ``physics``, ``properties``, ``initial_conditions``,
``postprocessor`` or ``solver`` adds its keys to those of the first, and a key
given again replaces its earlier value. A second call of
``boundary_conditions`` for the same boundary adds its keys to that boundary.
A second call of ``gravity``, ``time``, ``mesh_adaptivity`` or ``output``
replaces the whole section.

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
``dualmesh.input_error`` with a message that starts with the path of the key,
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
vocabulary is the file ``python/dualmesh/input/standard_names.toml``.

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

A formula holds these parts:

- A quantity in square brackets, such as ``[0.0375 m*K/W]`` or
  ``[2.77 eV]``, is a number with its unit, converted to SI.
- A bare number, such as ``2`` or ``1.5e-3``, is dimensionless.
- The operators are ``+``, ``-``, ``*``, ``/`` and ``^``. A product needs
  ``*``. The power ``^`` binds tightest and groups from the right, so
  ``2^3^2`` is :math:`2^9`, and a unary minus binds less tightly than ``^``,
  so ``-a^2`` is :math:`-(a^2)`. Parentheses group as usual.
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
   * - ``exp(a)``, ``log(a)``, ``sin(a)``, ``cos(a)``, ``tanh(a)``
     - the function of a dimensionless argument, with ``log`` the natural
       logarithm
   * - ``sqrt(a)``, ``abs(a)``
     - the square root and the absolute value
   * - ``max(a, b)``, ``min(a, b)``
     - the larger and the smaller of two values of the same dimension
   * - ``less(a, b)``, ``greater(a, b)``
     - the conditions :math:`a < b` and :math:`a > b`, for two values of
       the same dimension
   * - ``conditional(c, a, b)``
     - ``a`` where the condition ``c`` holds and ``b`` elsewhere, with ``c``
       made by ``less`` or ``greater`` and ``a`` and ``b`` of the same
       dimension

The input checks the dimension of every formula against the unit of its key.
A sum, ``max``, ``min``, ``less``, ``greater`` and the two branches of
``conditional`` need operands of the same dimension. A power of a quantity
with a dimension needs a number as its exponent. The number ``0`` matches
every dimension. The message of a formula of the wrong dimension names the
unit that the formula gives.

The formula language also parses ``time_step``, ``cell_size``, ``grad`` and
the tensor functions. The solve refuses them in a property or a boundary
value, which is a function of the fields, the position and the time.

.. _input-tables:

Tables
~~~~~~

A table gives a property as values on a grid of one to three inputs, such as
the temperature. It is a mapping with the one key ``table``, which holds the
columns inline or names a file of comma-separated values (CSV):

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

- The output column has the name of the property. Every other column is an
  input, whose name is a scalar that a formula reads, such as
  ``temperature``, ``time`` or ``x``.
- The table has one to three inputs.
- Every column holds the same number of values.
- The unit of each column has the dimension of its name.
- Each input holds at least two distinct values.
- The rows fill the full grid of the inputs, and no two rows hold the same
  inputs.

The solve refuses a table. Give a constant or a formula for a property that
the solve reads.

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
``boundary_conditions``, ``time``, ``mesh_adaptivity``, ``postprocessors``,
``output`` and ``solver``. Only ``mesh`` is required by the check of the
tree, and the solve also needs ``physics``.

.. _input-mesh:

mesh
~~~~

The section ``mesh`` holds exactly one of ``segment``, ``rectangle``, ``box``
and ``file``, and the optional key ``coordinate_system``.

A generated mesh starts at the origin and has linear cells. Its one region is
``domain``.

.. list-table:: The keys of a generated mesh
   :header-rows: 1
   :widths: 18 22 60

   * - Form
     - Key
     - Value
   * - ``segment``
     - ``length``
     - The length of the interval :math:`[0, \ell]` along :math:`x`, with
       :math:`\ell` the length in m. A positive length. Required.
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
A physics can also name it with its own key ``method``. A problem takes one
method, so the solve refuses two different methods.

Without a method, a problem with a ``fluid_dynamics`` of the ``equal_order``
formulation takes DMCDM, and every other problem takes FEM.

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

with :math:`I_C = \operatorname{tr}\mathbf{C}`. The first Piola-Kirchhoff
stress :math:`\mathbf{P} = \partial\psi/\partial\mathbf{F}_d` is the symbolic
derivative of the energy.

A solid at finite strain has these limits:

- It takes ``plane_strain`` or a three-dimensional mesh.
- It takes no ``time`` section and no initial displacement.
- It takes a ``traction`` on a boundary and refuses a ``pressure``, which
  would follow the deformed boundary.
- It takes no thermal expansion.
- Its ``Young_modulus`` and ``Poisson_ratio`` are one constant in every
  region.

.. code-block:: yaml

   physics:
     solid_mechanics:
       formulation: plane_strain
       kinematics: finite
       strain_energy: "shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + first_Lame_parameter/2*log(J)^2"

.. rubric:: fluid_dynamics

``fluid_dynamics`` solves the balance of mass and momentum of an
incompressible Newtonian fluid for its velocity :math:`\mathbf{v}` in :math:`\mathrm{m/s}` and
its pressure :math:`p` in Pa.

``formulation``
   ``equal_order`` or ``Taylor-Hood``. Default ``equal_order``, the velocity
   and the pressure of the same order with a pressure stabilisation, which
   runs with both methods and on the linear cells of a generated mesh.
   ``Taylor-Hood`` takes a quadratic velocity and a linear pressure. It needs
   ``method: FEM`` and a mesh file of quadratic cells.

.. code-block:: yaml

   physics:
     fluid_dynamics:
       formulation: equal_order

.. rubric:: Couplings

The input couples its physics from the physics and the properties it holds:

- ``heat_conduction`` with ``solid_mechanics`` and the properties
  ``linear_expansion_coefficient`` and ``stress_free_temperature`` gives a
  thermal expansion of the solid. The solve takes the two properties
  together, and refuses them without ``heat_conduction`` or at finite strain.
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
     - ``heat_conduction`` in a transient or with a flow, ``solid_mechanics``
       in a transient or under gravity, ``fluid_dynamics`` always
     - per region for ``heat_conduction``, one constant for
       ``solid_mechanics`` and ``fluid_dynamics``
   * - ``specific_heat_capacity``
     - ``J/(kg*K)``
     - ``heat_conduction`` in a transient or with a flow
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
     - one constant
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

"Per region" means a constant or a formula, in every region or one per region.
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

The section ``models`` holds balance laws of your own, one entry per law.
The solve refuses a problem with a ``models`` section, so a problem takes its
equations from ``physics``.

An entry states the law

.. math::

   \frac{\partial m}{\partial t} + \frac{\partial^2 a}{\partial t^2}
   + \nabla\cdot\mathbf{F} = s

for its unknown, with :math:`m` the storage, :math:`a` the inertia,
:math:`\mathbf{F}` the flux and :math:`s` the source. Its keys are:

``unknown``
   The field the law solves for: a name, or a mapping of one name to its
   ``unit`` and ``shape``. Required.

``expressions``
   Named formulas that the terms read. A name outside the vocabulary takes
   the form ``{unit: ..., value: ...}``.

``states``
   Internal states, each with ``unit``, ``initial``, ``shape`` and its
   ``rate``, the formula of its time derivative. ``rate`` is required.

``storage``, ``inertia``, ``flux``, ``source``
   The formulas of :math:`m`, :math:`a`, :math:`\mathbf{F}` and :math:`s`.

``boundary_fluxes``
   Named formulas of the outward normal flux :math:`\mathbf{F}\cdot\mathbf{n}`,
   with :math:`\mathbf{n}` the outward unit normal. A positive boundary flux
   removes the conserved quantity.

``flux_name``
   The standard name of :math:`\mathbf{F}`, which a ``boundary_flux``
   postprocessor names.

``method``
   ``FEM`` or ``DMCDM``.

.. _input-initial-conditions:

initial_conditions
~~~~~~~~~~~~~~~~~~

The section ``initial_conditions`` gives the value of each field at the start
of a transient run, by its standard name. A scalar field takes one constant.
A vector field takes a list of one constant per direction. A formula and a
table are refused.

- A transient ``heat_conduction`` needs the initial ``temperature``.
- An initial ``displacement`` needs the ``time`` section, because it starts a
  run with inertia. A solid at finite strain refuses it.
- A flow takes an initial ``velocity`` and ``pressure``.
- A field that no physics of the problem solves for is refused, and so is a
  component name such as ``displacement_x``.

.. code-block:: yaml

   initial_conditions:
     temperature: 573 K

.. _input-boundary-conditions:

boundary_conditions
~~~~~~~~~~~~~~~~~~~

The section ``boundary_conditions`` maps each boundary to the conditions on
it. Each key sets a field or a flux through the boundary. A boundary of a
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
     - A list of one constant per direction, in m.
   * - ``displacement_x``, ``displacement_y``, ``displacement_z``
     - ``solid_mechanics``
     - One component in m. The other components stay free.
   * - ``traction``
     - ``solid_mechanics``
     - The force per area, a list of one constant per direction, in Pa. At
       finite strain it is :math:`\mathbf{P}\mathbf{N}` per undeformed area,
       with :math:`\mathbf{N}` the outward unit normal of the undeformed
       boundary.
   * - ``pressure``
     - ``solid_mechanics`` in a problem without a flow
     - The pressure :math:`p` in Pa, the load :math:`-p\,\mathbf{n}`.
   * - ``velocity``
     - ``fluid_dynamics``
     - A list of one constant per direction, in :math:`\mathrm{m/s}`.
   * - ``velocity_x``, ``velocity_y``, ``velocity_z``
     - ``fluid_dynamics``
     - One component in :math:`\mathrm{m/s}`.
   * - ``pressure``
     - ``fluid_dynamics``
     - The pressure in Pa.

In the table, :math:`h_c` is the heat transfer coefficient in
:math:`\mathrm{W/(m^2\,K)}`, :math:`\epsilon` the dimensionless emissivity
and :math:`\sigma_\mathrm{SB} = 5.670374419 \times 10^{-8}\,\mathrm{W/(m^2\,K^4)}`
the Stefan-Boltzmann constant. In axisymmetric coordinates the components are
``displacement_r``, ``displacement_z``, ``velocity_r`` and ``velocity_z``.

A scalar value, and each constant of ``convection`` and ``radiation``, is a
constant or a formula. A list takes constants only, so give a formula per
component with the component key, such as ``velocity_x``.

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
   names the controller, with the one key of that controller. Default
   ``{by: error, relative_tolerance: 1e-3}``.

.. list-table:: The step controllers
   :header-rows: 1
   :widths: 16 34 50

   * - ``by``
     - Key and default
     - Step
   * - ``error``
     - ``relative_tolerance``, a number above 0 and at most 1. Default
       :math:`10^{-3}`.
     - The step that keeps the estimated local error of each variable,
       against its range, at the tolerance. A step whose error exceeds it is
       repeated with a smaller step.
   * - ``iterations``
     - ``target_iterations``, an integer of at least 1. Default 5.
     - A larger step after a step that took fewer nonlinear iterations than
       the target, and a smaller step after one that took more.
   * - ``Courant``
     - ``Courant_number``, a positive number. Default 0.5.
     - The step that gives the Courant number of a flow.
   * - ``fixed``
     - no key
     - The initial time step throughout, with the last step shortened so
       that the run ends at ``end_time``.

The default controller ``error`` sets the step by the accuracy in time. A key
of another controller is refused, such as ``relative_tolerance`` with
``by: fixed``. The solve refuses ``by: Courant`` for every input, because the
Courant controller needs the projection time integration of a flow, which the
input does not select.

The time scheme is backward Euler, and a solid with inertia takes the
generalised-:math:`\alpha` method [ChungHulbert1993]_.
:doc:`/user_guide/solving` describes the transient solve.

.. code-block:: yaml

   time:
     end_time: 40 s
     initial_time_step: 0.1 s
     step_control: {by: fixed}

.. _input-mesh-adaptivity:

mesh_adaptivity
~~~~~~~~~~~~~~~

The section ``mesh_adaptivity`` refines the mesh where the error indicator of
a field is large, and coarsens it where the indicator is small. Every key is
required.

``indicator``
   ``gradient_recovery``, the indicator of Zienkiewicz and Zhu
   [ZienkiewiczZhu1987]_, from the difference between the gradient of the
   field and its recovered gradient on each cell. For a vector field the
   squared indicators of its components add up.

``field``
   The field whose indicator drives the adaptivity: a field that a physics
   of the problem solves for, such as ``temperature``.

``refine_fraction``
   A number from 0 to 1. The cells refined are the smallest set whose squared
   indicators add up to this fraction of their total, the largest first
   [Dorfler1996]_.

``coarsen_fraction``
   A number from 0 to 1. The cells coarsened are the largest set of refined
   cells, outside those marked for refining, whose squared indicators add up
   to at most this fraction of the total, the smallest first.

``interval``
   The number of accepted time steps between two adaptations of a transient
   run. An integer of at least 1. A steady run reads no interval, and the key
   is still required.

``maximum_refinement_level``
   The number of times a cell of the starting mesh can be refined. An
   integer of at least 0.

A steady run solves, marks and adapts in cycles. Each solve starts from the
solution of the last mesh. The cycles stop when no marked cell can be refined
further, or after :math:`4 d\,n_\mathrm{max} + 1` cycles, with :math:`d` the
dimension of the mesh and :math:`n_\mathrm{max}` the maximum refinement
level. A transient run adapts
the mesh after every ``interval`` accepted steps and goes on from the time
reached, with the solution carried to the new mesh and the step that the
controller chose. :doc:`/user_guide/adaptivity` describes adaptive
refinement.

.. code-block:: yaml

   mesh_adaptivity:
     indicator: gradient_recovery
     field: temperature
     refine_fraction: 0.5
     coarsen_fraction: 0.05
     interval: 1
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
       plane problem and :math:`\mathrm{W/m^2}` on a segment.

In axisymmetric coordinates an integral spans the body of revolution. The
field of a reduction is a
scalar field, such as ``temperature`` or ``pressure``, or a component, such as
``displacement_x``. A ``boundary_flux`` is positive when the heat leaves the
body.

The solve has these limits:

- ``boundary_flux`` takes ``heat_flux`` in a problem with
  ``heat_conduction``.
- A stress is a ``point_value`` of one component, such as ``stress_yy``, in a
  steady solid at small strain in Cartesian coordinates, with ``Young_modulus``
  and ``Poisson_ratio`` one constant in every region. It includes the thermal
  strain of a thermal expansion. A point outside the mesh is refused.
- The check accepts a ``region`` key on ``integral``, ``average``,
  ``maximum`` and ``minimum``, and the solve refuses it.

.. code-block:: yaml

   postprocessors:
     heat_through_left: {boundary_flux: heat_flux, boundary: left}
     temperature_bottom: {point_value: temperature, point: [0.05 m, 0 m]}
     hottest: {maximum: temperature}

.. _input-output:

output
~~~~~~

The section ``output`` writes the fields and the postprocessors to files. All
three keys are required.

``file``
   The base name of the files, relative to the folder of the case file, or
   to the working directory in Python.

``interval``
   The fields are written every ``interval``-th accepted time step and at the
   end of the run. An integer of at least 1. A steady run writes one file.

``fields``
   The list of fields to write: ``temperature``, ``displacement``,
   ``velocity`` and ``pressure``, those of the problem.

The run writes these files:

- ``<file>_0000.vtu``, ``<file>_0001.vtu`` and so on, one VTK (Visualization
  Toolkit) unstructured grid file per output time,
- ``<file>.pvd``, a collection of those files, which ParaView opens as one
  time series,
- ``<file>.csv``, the postprocessors, one column per postprocessor after the
  column ``time [s]``, with the unit in each header, as in ``peak [K]``. The
  run writes it when the input has postprocessors.

.. code-block:: yaml

   output:
     file: bus_bar
     interval: 1
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
   ``true`` and ``false`` become the texts ``true`` and ``false``. Options
   that start with ``snes_`` go to the nonlinear solver of PETSc, and the
   others replace the choice of the linear solver. A leading dash is refused.

The solve refuses ``relative_tolerance`` together with
``PETSc_options.snes_rtol``, which set the same tolerance. Keep
``relative_tolerance``. :doc:`/user_guide/solving` describes the options and
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
     interval: 1
     fields: [temperature]

The test ``tests/python/test_front_end.py`` checks that the temperature at the
middle of the bottom edge is :math:`83.142\,^\circ\mathrm{C}`, the value of
the book to three decimals.
