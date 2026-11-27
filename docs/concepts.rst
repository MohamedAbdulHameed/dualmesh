Concepts
========

A simulation in dualmesh is a *problem*: a mesh, the physics that
act on it, the properties of the materials, the initial and boundary
conditions, and a study that solves them.  This page defines each part in that
order, then the two methods that discretize the equations, and then the object
level, which builds equations term by term.  :doc:`quickstart` solves a first
problem, and :doc:`user_guide/input` lists every key.

The problem
-----------

A problem holds one input tree.  You write the tree as a YAML or TOML case file
and solve it with ``dualmesh run case.yaml``, or you build it in Python with
:class:`dualmesh.problem` and solve it with :meth:`~dualmesh.problem.solve`.
Each top-level section of the tree has one Python call:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Section
     - Python call
     - Content
   * - ``mesh``
     - ``dualmesh.problem(mesh=...)``
     - The mesh and its coordinate system.  Required.
   * - ``method``
     - ``dualmesh.problem(mesh=..., method=...)``
     - ``FEM`` or ``DMCDM``.
   * - ``physics``
     - ``physics(name, ...)``
     - The physics and their options.
   * - ``properties``
     - ``properties(...)``
     - The properties of the materials, by standard name.
   * - ``gravity``
     - ``gravity([...])``
     - The gravitational acceleration :math:`\mathbf{g}` in
       :math:`\mathrm{m/s^2}`.
   * - ``models``
     - ``model(name, ...)``
     - Balance laws of your own.
   * - ``initial_conditions``
     - ``initial_conditions(...)``
     - The value of each field at the start of a transient run.
   * - ``boundary_conditions``
     - ``boundary_conditions(boundary, ...)``
     - The conditions on each named boundary.
   * - ``time``
     - ``time(...)``
     - The time span and the time step control of a transient run.
   * - ``eigenvalues``
     - ``eigenvalues(...)``
     - The number of eigenvalues of an eigenvalue study and the value they
       are closest to.
   * - ``mesh_adaptivity``
     - ``mesh_adaptivity(...)``
     - The adaptive refinement of the mesh.
   * - ``postprocessors``
     - ``postprocessor(name, ...)``
     - Named scalar results, and the values of a field along a line.
   * - ``output``
     - ``output(file, fields, interval)``
     - The files of the fields and the post-processors.
   * - ``solver``
     - ``solver(...)``
     - The relative tolerance of the nonlinear solve and options of PETSc, the
       Portable, Extensible Toolkit for Scientific Computation.

Every quantity is a text of a number and its unit, such as ``"45 W/(m*K)"``,
``"40 degC"`` or ``"1 MW/m^3"``.  The input checks the dimension of each unit
and converts each value to the International System of Units (SI).  A number
without a unit is dimensionless, so ``Poisson_ratio=0.3`` needs no unit.

The mesh
--------

The mesh section holds exactly one of four forms:

* ``segment``, with its ``length`` and its number of ``cells``.  Its boundaries
  are ``left`` and ``right``.
* ``rectangle``, with its ``size`` and its ``cells`` in each direction, and the
  ``cell_type`` ``quadrilateral`` (the default) or ``triangle``.  Its boundaries
  are ``left``, ``right``, ``bottom`` and ``top``.
* ``box``, with its ``size`` and its ``cells`` in each direction, and the
  ``cell_type`` ``hexahedron`` (the default) or ``tetrahedron``.  Its
  boundaries are those of the rectangle and ``back`` and ``front``.
* ``file``, the path of a mesh file, with the ``length_unit`` of its
  coordinates, such as ``mm``.  The file names its boundaries and its regions.

A generated mesh has linear cells and one region, ``domain``.  Its ``origin``
is the corner with the smallest coordinates, at 0 m by default.  Its
``grading`` is the ratio of the length of each cell to the length of the cell
before it along each axis, 1 by default, which gives a uniform mesh.  The ``coordinate_system`` is ``cartesian`` (the default) or
``axisymmetric``, which takes a mesh of space dimension 2 with the radius
:math:`r` as its first coordinate and the axis :math:`z` as its second.
:doc:`user_guide/meshes` describes the mesh files, and the mesh generators of
the object level.

The physics
-----------

A physics is a set of balance laws with its fields.  Each one has the form

.. math::

   \frac{\partial m}{\partial t} + \nabla \cdot \mathbf{F} = s,

where :math:`m` is the stored quantity per volume, :math:`\mathbf{F}` its flux
and :math:`s` its source per volume.  For heat conduction, the stored heat per
volume is :math:`m = \rho c_p T`, the flux is the heat flux
:math:`\mathbf{q} = -k \nabla T` in :math:`\mathrm{W/m^2}`, and :math:`s` is
the heat source in :math:`\mathrm{W/m^3}`.  Here :math:`T` is the temperature
in :math:`\mathrm{K}`, :math:`\rho` the density in :math:`\mathrm{kg/m^3}`,
:math:`c_p` the specific heat capacity in :math:`\mathrm{J/(kg\,K)}` and
:math:`k` the thermal conductivity in :math:`\mathrm{W/(m\,K)}`.

The input has three physics, and its ``models`` section adds balance laws of
your own (:doc:`tutorials/user_model`).  Each physics needs some properties,
and reads others when you give them:

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32

   * - Physics
     - Fields
     - Properties it needs
     - Properties it reads when given
   * - ``heat_conduction``
     - ``temperature``
     - ``thermal_conductivity``, and ``density`` and
       ``specific_heat_capacity`` in a transient run or with a flow
     - ``volumetric_heat_source``
   * - ``solid_mechanics``
     - ``displacement``
     - ``Young_modulus`` and ``Poisson_ratio``, and ``density`` in a transient
       run or under gravity
     - ``body_force_density``, ``linear_expansion_coefficient``,
       ``stress_free_temperature`` and ``yield_stress``
   * - ``fluid_dynamics``
     - ``velocity`` and ``pressure``
     - ``density`` and ``dynamic_viscosity``, and
       ``volumetric_expansion_coefficient`` and ``reference_temperature`` for
       a heated flow under gravity
     - ``body_force_density``

The options of ``solid_mechanics`` are its ``formulation``, ``plane_strain`` or
``plane_stress`` on a plane mesh, its ``kinematics``, ``small`` (the default)
or ``finite`` strain, and the ``strain_energy`` of a finite strain:
``neo-Hookean`` (the default), ``Saint_Venant_Kirchhoff`` or a formula of the
invariants of the deformation.  In three dimensions and in axisymmetric
coordinates, the mesh sets the formulation.  A ``yield_stress`` makes a solid
at small strain plastic.

The option of ``fluid_dynamics`` is its ``formulation``: ``equal_order``, the
same interpolation of the velocity and the pressure with a pressure
stabilization, ``Taylor-Hood``, a quadratic velocity and a linear pressure on
a quadratic mesh with FEM, or ``penalty``, which eliminates the pressure by a
penalty on the divergence of the velocity.  Without the option, a flow takes
``Taylor-Hood`` with FEM on a mesh of quadratic cells, and ``equal_order``
otherwise.

Two physics of one problem couple without a further section:

* ``heat_conduction`` with ``solid_mechanics`` and the properties
  ``linear_expansion_coefficient`` and ``stress_free_temperature`` gives the
  thermal strain of the solid.
* ``heat_conduction`` with ``fluid_dynamics`` carries the heat with the flow.
  Under gravity, the fluid also takes the Boussinesq force
  :math:`-\rho \beta (T - T_0) \mathbf{g}`, where :math:`\beta` is the
  volumetric expansion coefficient in :math:`\mathrm{1/K}` and :math:`T_0` the
  reference temperature in :math:`\mathrm{K}`.

The ``gravity`` section gives one component of :math:`\mathbf{g}` per
direction of space, and solids and flows take its force per volume
:math:`\rho \mathbf{g}`.

The properties
--------------

A property has a standard name, such as ``thermal_conductivity``, and the
standard name fixes its unit.  A property takes a constant, a formula or a
table, or on a mesh file one value per region.  The input refuses a property that no physics
of the problem reads, and it names a property that a physics needs and the
input leaves out.

The conditions
--------------

The ``initial_conditions`` section gives a constant or a formula of the
coordinates for each field at the start of a transient run, and a list of one
value per component for a vector field.

The ``boundary_conditions`` section holds one entry per boundary.  The word
``symmetry`` in place of the entry makes the boundary a plane of symmetry.
Otherwise each key of the entry sets either a field or a flux, with a constant
or a formula:

* A field, such as ``temperature``, ``displacement_x`` or ``velocity``, takes its
  value on the boundary.  A vector field takes one value per component.
* ``heat_flux`` takes the heat flux :math:`\mathbf{q} \cdot \mathbf{n}` that
  leaves through the boundary, with :math:`\mathbf{n}` the outward unit normal.
* ``convection`` takes a ``heat_transfer_coefficient`` and an
  ``ambient_temperature``, and ``radiation`` takes an ``emissivity`` and an
  ``ambient_temperature``.
* ``traction`` takes the force per area on a solid, one component per
  direction, and ``pressure`` on a solid takes the load :math:`-p\,\mathbf{n}`,
  with :math:`p` the pressure.

A boundary without a condition has no flux through it: it is insulated for heat
conduction and free of traction for a solid.

The studies
-----------

The sections of the input choose the study:

* Without a ``time`` section, the study is steady.
* A ``time`` section makes the study transient from the time 0 s to its
  ``end_time``, starting with its ``initial_time_step``.  Its ``step_control``
  sets the time step :math:`\Delta t` by the estimated error of the step
  (``error``, the default) or by the number of Newton iterations of the step
  (``iterations``), or keeps it ``fixed``.
* An ``eigenvalues`` section makes the study an eigenvalue study, which finds
  the ``count`` eigenvalues closest to ``near``.
* A ``mesh_adaptivity`` section refines and coarsens the mesh where the
  estimated error of a field is largest.  A steady study repeats the solve on
  the adapted mesh, and a transient study adapts the mesh every ``interval``
  accepted time steps.

The ``postprocessors`` section defines named results of the solution: the
``integral``, ``average``, ``maximum`` or ``minimum`` of a field, its
``point_value`` at a point, the ``boundary_flux`` through a boundary, or its
values along a ``line`` between two points.  The
``output`` section writes the fields and the post-processors to files.
:doc:`user_guide/solving` describes the solvers and the parallel runs.

The two methods
---------------

The unknowns of both methods are the values of the fields at the nodes of the
mesh, and each field :math:`u` is interpolated between them by the shape
functions :math:`N_i` of the elements.  The methods differ in the statement of
the balance law that gives one equation per node :math:`i`.

The finite element method (FEM) weights the balance law with :math:`N_i` and
integrates over the domain :math:`\Omega`:

.. math::

   \int_\Omega N_i \frac{\partial m}{\partial t} \,\mathrm{d}V
   - \int_\Omega \nabla N_i \cdot \mathbf{F} \,\mathrm{d}V
   + \oint_{\partial\Omega} N_i \, \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S
   = \int_\Omega N_i \, s \,\mathrm{d}V.

The dual mesh control domain method (DMCDM) [Reddy2024]_ integrates the
balance law over the control domain :math:`V_i` of node :math:`i`.  In each
element around the node, the lines that join the midpoints of the edges to the
centroid of the element cut off the part of the element that holds the node.
In three dimensions, surfaces through the midpoints of the edges, the
centroids of the faces and the centroid of the element cut it off.
:math:`V_i` joins these parts, and the control domains of all the nodes form
the dual mesh.  The balance over :math:`V_i` is

.. math::

   \int_{V_i} \frac{\partial m}{\partial t} \,\mathrm{d}V
   + \oint_{\partial V_i} \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S
   = \int_{V_i} s \,\mathrm{d}V,

with :math:`\mathbf{n}` the outward unit normal of :math:`\partial V_i`.  The
finite element interpolation gives the flux on each face of a control domain.
The flux through a face shared by two control domains leaves one and enters the
other, so DMCDM conserves the balanced quantity in every control domain.

Without a ``method``, a problem with a ``fluid_dynamics`` of the
``equal_order`` formulation takes DMCDM, with any solid or heat conduction
beside the flow, and every other problem takes FEM.
The ``method`` section, or a ``method`` key of a physics or of a model, chooses the method,
and one problem takes one method.  DMCDM refuses the serendipity elements
``Quad8`` and ``Hex20`` and the pyramid ``Pyramid5``, for which the code builds
no dual mesh.  :doc:`theory/foundations` derives both statements.


The object level
----------------

The input builds its problem from the objects of the code, which you can also
use directly.  :class:`dualmesh.Problem` takes a mesh and a method.  Its
:meth:`~dualmesh.Problem.add_physics` adds a physics, such as
``heat_conduction``, ``solid_mechanics``, ``fluid_dynamics`` or
``general_form_PDE``, and :meth:`~dualmesh.Problem.add_kernel` adds one term of
an equation.  A kernel supplies a flux, a source, or both, to the equation of
one field, and a property object computes a named property for the kernels.
The object level adds studies that the input does not have, such as
:meth:`~dualmesh.Problem.solve_buckling`.  :doc:`user_guide/problem_setup`
describes it.
