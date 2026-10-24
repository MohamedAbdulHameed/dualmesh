The command line
================

The ``dualmesh`` command runs a case file, lists and describes the registered
types, and checks the side sets of a mesh file:

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

A case file is the physics level of a problem written as data
(:doc:`/user_guide/case_files`).

.. currentmodule:: dualmesh

.. autosummary::
   :toctree: generated/

   run_case
   load_case
   Case
