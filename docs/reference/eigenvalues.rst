Eigenvalue and buckling studies
===============================

.. currentmodule:: dualmesh.modes

An eigenvalue study computes the eigenvalues and the eigenvectors of the
operator of a problem (:meth:`dualmesh.Problem.solve_eigenvalue`), such as the
vibration modes of a structure.  A buckling study computes the load factors
at which a solved structure buckles, and its buckling modes
(:meth:`dualmesh.Problem.solve_buckling`).

.. autosummary::
   :toctree: generated/

   solve_modes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EigenmodesResult
   BucklingResult
