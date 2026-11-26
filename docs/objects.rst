Object reference
================

Each kernel, boundary condition, nodal load, property object and constraint
is a registered type with declared parameters.  When you add an object, the
code checks every parameter you give against these declarations.  The pages of
this reference are generated from the same declarations when the docs are
built, so each page lists the parameters that the code accepts.  The reference
also has a page for each physics and each coupling.

Each page gives the purpose of the type, its required parameters, and its
optional parameters with their defaults.  Most pages end with an example.

Parameters from the base classes
--------------------------------

The types of one category share a base class, and the base class declares the
parameters below.  Each page lists them with the parameters of its own type.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Parameters of the base class
   * - Kernel
     - ``variable`` (required), ``block``, ``quadrature``,
       ``reduced_integration``
   * - Boundary condition
     - ``variable`` and ``boundary`` (required), ``block``, ``quadrature``,
       ``reduced_integration``
   * - Nodal boundary condition
     - ``variable`` and ``boundary`` (required)
   * - Nodal load
     - ``variable`` (required)
   * - Property
     - ``block``, ``property_factors``
   * - Constraint
     - none

``variable`` names the variable whose equation the object acts on.  A list of
variables creates the object once for each of them, with the same values of
the other parameters.  ``boundary`` names the side sets or node sets where the
condition applies.  :doc:`/user_guide/problem_setup` explains ``block``,
``quadrature``, ``reduced_integration`` and ``property_factors``.

Three types depart from the table:

- The boundary condition ``gap_heat_transfer`` derives from the base class of
  interface conditions, which adds ``secondary_boundary`` (required) and
  ``secondary_variable``.
- The nodal boundary condition ``point_Dirichlet`` takes a ``point`` in place of
  ``boundary``.
- Each constraint declares all of its parameters.  ``periodic`` takes an
  optional ``variable``, and ``uniform_field`` a required one.

Look up a type from Python
--------------------------

The registry gives the same information in Python:

.. code-block:: python

   import dualmesh

   dualmesh.registered_types()  # every object type
   dualmesh.object_category("diffusion")  # 'kernel'
   print(dualmesh.describe_object("diffusion"))

On the command line, ``dualmesh list`` prints the types and
``dualmesh describe diffusion`` prints the parameters of one type.

.. toctree::
   :maxdepth: 1

   syntax/index
