The command line
================

The ``dualmesh`` command runs a case file, lists and describes the registered
types, and lists the side sets of a mesh file:

.. code-block:: console

   dualmesh run CASE.yaml
   dualmesh list [--category CATEGORY] [--module MODULE]
   dualmesh describe TYPE
   dualmesh mesh FILE [--sidesets OUTPUT.vtu]
   dualmesh --version

.. currentmodule:: dualmesh.cli

.. autosummary::
   :toctree: generated/

   main

Case files
----------

A case file is a YAML (YAML Ain't Markup Language) or TOML (Tom's Obvious
Minimal Language) file of the input sections: the mesh, the physics, the
properties, the conditions, the time span and the settings of the solve.
:doc:`/user_guide/input` describes every section and key.  ``dualmesh run``
reads the file with :meth:`dualmesh.problem.from_file` and solves it with
:meth:`dualmesh.problem.solve`.
