Uncertainty quantification
==========================

.. currentmodule:: dualmesh.uq

The propagation of uncertain inputs through a model, sensitivity indices,
Gaussian process surrogates and Bayesian calibration.  The theory and its
verification are in :doc:`/theory/uncertainty`.

Probability distributions
-------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Normal
   LogNormal
   Uniform
   LogUniform
   Distribution

Propagation
-----------

.. autosummary::
   :toctree: generated/

   propagate
   wilks_samples

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Runs

Sensitivity
-----------

.. autosummary::
   :toctree: generated/

   sobol

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SobolIndices

Surrogates
----------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GaussianProcess

Calibration
-----------

.. autosummary::
   :toctree: generated/

   calibrate

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Posterior
