Meshes
======

.. currentmodule:: dualmesh

A :class:`Mesh` is the primal mesh of finite elements: its nodes, its
elements in blocks, and the side sets and node sets on which boundary
conditions act.  You generate a mesh, read it from a file, or build it
from arrays.

The mesh
--------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Mesh

Generating a mesh
-----------------

.. autosummary::
   :toctree: generated/

   generate_line_mesh
   generate_rectangle_mesh
   generate_box_mesh
   generate_annulus_mesh
   mesh_from_arrays

Generating a surface
--------------------

.. autosummary::
   :toctree: generated/

   generate_cylinder_surface_mesh
   generate_spherical_shell_mesh
   generate_surface_mesh

Graded coordinates
------------------

.. autosummary::
   :toctree: generated/

   graded_coordinates
   meshing.coordinates_from_spacings

Reading, writing and inspecting
-------------------------------

.. autosummary::
   :toctree: generated/

   read_mesh
   write_mesh
   write_sidesets
   sideset_summary
