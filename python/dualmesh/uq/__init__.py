# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Uncertainty quantification and sensitivity analysis.

A model is any Python function of keyword arguments that returns a number,
an array, or a dict of them.  The uncertain inputs are distributions::

    from dualmesh import uq

    inputs = {"k": uq.Normal(1.0, 0.05), "D": uq.LogNormal(median=1.0, factor=10)}
    runs = uq.propagate(model, inputs, samples=200)
    print(runs.summary())
    indices = uq.sobol(model, inputs, samples=512)
    posterior = uq.calibrate(model, inputs, observed={"T": T_measured}, noise={"T": 20.0})
    print(posterior.summary())

See the documentation chapter on uncertainty quantification for the theory,
the sources and the verification.
"""

from .calibrate import Posterior, calibrate
from .distributions import Distribution, LogNormal, LogUniform, Normal, Uniform
from .gaussian_process import GaussianProcess
from .propagate import Runs, propagate, wilks_samples
from .sobol import SobolIndices, sobol

__all__ = ["Distribution", "GaussianProcess", "LogNormal", "LogUniform", "Normal", "Posterior", "Runs", "SobolIndices", "Uniform", "calibrate", "propagate", "sobol", "wilks_samples"]
