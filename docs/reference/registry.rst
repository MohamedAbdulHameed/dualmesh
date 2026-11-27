The object registry
===================

.. currentmodule:: dualmesh

The registry holds the names, the categories and the parameters of the
registered types: the physics, the couplings and the objects of
:doc:`/objects`.

.. autosummary::
   :toctree: generated/

   list_objects
   describe
   registered_types
   object_category
   object_module
   describe_object

A misspelled name, a missing parameter or a value outside its range raises
:class:`InputError`, from the C++ core and from the Python input alike.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   InputError

Parameter declarations
----------------------

The physics and the couplings declare their parameters as dataclass fields
made by :func:`parameter`, each with a unit and a description.  The same
declaration checks the keywords and prints the reference of a type.

.. currentmodule:: dualmesh.parameters

.. autosummary::
   :toctree: generated/

   parameter
   keyword_checked
   check_keywords
   describe_fields
