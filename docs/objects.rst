Object reference
================

Every kernel, boundary condition, material and nodal load in dualmesh is a
registered object with declared, validated parameters.  This part of the manual
has one page for each of them, generated from that registry when the
documentation is built, so a page can never describe a parameter the object
does not have.

Each page gives the object's purpose, its required parameters, its optional
parameters with their defaults, and prose on how the object is used and what
its keywords mean.  The four parameters every object accepts — ``block``,
``quadrature``, ``reduced_integration`` and ``scale_with_load`` — are explained
once, in :doc:`/user_guide/problem_setup`, rather than repeated on every page.

The same information is available at the prompt:

.. code-block:: python

   import dualmesh as dm

   dm.registered_types()                    # every object name
   dm.object_category("heat_conduction")     # 'kernel'
   print(dm.describe_object("heat_conduction"))

.. toctree::
   :maxdepth: 1

   syntax/index
