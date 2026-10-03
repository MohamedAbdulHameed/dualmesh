.. _uncertainty:

Uncertainty quantification and sensitivity analysis
===================================================

A calculated result agrees with a measurement in a meaningful sense only when
the difference between them is compared with the uncertainty of both.  The
uncertainty of a calculation comes from its inputs (the power history, the
dimensions), from the parameters of its models (a diffusion coefficient known
to a factor of ten) and from the models themselves.  The package
:mod:`dualmesh.uq` propagates the uncertainty of inputs and parameters to the
results of any dualmesh calculation, ranks the parameters by their influence,
and calibrates them against measurements.  It works with any model: a Python
function of keyword arguments that returns a number, an array, or a dict of
them.

The package has five parts: the distributions of the inputs, forward
propagation (:func:`~dualmesh.uq.propagate`), variance-based sensitivity
(:func:`~dualmesh.uq.sobol`), a Gaussian process surrogate
(:class:`~dualmesh.uq.GaussianProcess`) and Bayesian calibration
(:func:`~dualmesh.uq.calibrate`).  A fuel rod exposes its uncertain models
through one input group, :class:`~dualmesh.fuel.ModelFactors`.

A short example::

    from dualmesh import uq

    def bus_bar(conductivity, heat_source):
        ...  # build and solve the problem
        return {"hottest": T_max}

    inputs = {"conductivity": uq.Normal(20.0, 1.0), "heat_source": uq.LogNormal(median=1.0e6, factor=1.2)}
    runs = uq.propagate(bus_bar, inputs, samples=200)
    print(runs.summary())
    indices = uq.sobol(bus_bar, inputs, samples=512)
    print(indices.summary())

The complete example is ``examples/bus_bar_uq.py``.

Distributions of the inputs
---------------------------

Each input is a distribution with an inverse cumulative distribution
function :math:`F^{-1}`, so that a design of the unit cube, :math:`u_j \in
(0, 1)`, becomes a sample :math:`x_j = F_j^{-1}(u_j)` of the inputs.

``Normal(mean, standard_deviation, lower, upper)``
    The normal distribution :math:`N(\mu, \sigma^2)`.  With a bound it is
    the normal variable conditioned on :math:`[a, b]`, whose mean and
    variance are

    .. math::

        E[X] = \mu + \sigma \frac{\phi(\alpha) - \phi(\beta)}{Z}, \qquad
        \mathrm{Var}[X] = \sigma^2\left[1 + \frac{\alpha\phi(\alpha) -
        \beta\phi(\beta)}{Z} - \left(\frac{\phi(\alpha) -
        \phi(\beta)}{Z}\right)^2\right],

    with :math:`\alpha = (a - \mu)/\sigma`, :math:`\beta = (b - \mu)/\sigma`,
    :math:`Z = \Phi(\beta) - \Phi(\alpha)`, and :math:`\phi`, :math:`\Phi`
    the standard normal density and distribution function.  A parameter
    that must stay positive, such as a grain radius factor of standard
    deviation 0.3, is given ``lower=0``.

``LogNormal(median, factor, sigma, lower, upper)``
    :math:`\ln X \sim N(\ln m, \sigma^2)`.  The spread is given either as
    :math:`\sigma` or as the factor :math:`f` of the central 95 % interval
    :math:`[m/f, m f]`, so that :math:`\sigma = \ln f / 1.95996`.  A
    diffusion coefficient uncertain by a factor of 100 in all, from 0.1 to 10
    times the reference value, is ``LogNormal(median=1, factor=10)``.  Its
    moments are :math:`E[X^n] = m^n e^{n^2\sigma^2/2} [\Phi(\beta - n\sigma) -
    \Phi(\alpha - n\sigma)]/Z`, with :math:`\alpha` and :math:`\beta` the
    bounds of :math:`\ln X` in units of :math:`\sigma`.

``Uniform(lower, upper)`` and ``LogUniform(lower, upper)``
    Uniform in :math:`X` or in :math:`\ln X`.

Forward propagation
-------------------

:func:`~dualmesh.uq.propagate` runs the model at ``samples`` points of the
input space and returns :class:`~dualmesh.uq.Runs`, which holds every input
and output and computes their statistics.  Three designs are available.

* ``latin_hypercube`` (default): the range of each input is divided into
  :math:`n` intervals of equal probability, each interval receives one run,
  and the intervals of different inputs are paired at random (McKay,
  Beckman and Conover [McKay1979]_).  For outputs that depend mostly on one
  input at a time, the error of the mean is much smaller than that of random
  sampling at the same cost.
* ``sobol``: a scrambled Sobol' sequence, whose error of the mean falls
  almost as :math:`n^{-1}` for smooth models, against :math:`n^{-1/2}` for
  random sampling.  It is best used with :math:`n` a power of 2.
* ``monte_carlo``: independent random runs.  The Wilks limits below need
  them.

The runs can execute in parallel (``processes``), and a store file
(``store``) keeps each run as it finishes.  An interrupted study then
resumes where it stopped, and later studies reuse the runs with the same
input values.  A run that fails is recorded with its error message and
excluded from the statistics.

Statistics
~~~~~~~~~~

:meth:`~dualmesh.uq.Runs.mean`, :meth:`~dualmesh.uq.Runs.standard_deviation`,
:meth:`~dualmesh.uq.Runs.percentile` and :meth:`~dualmesh.uq.Runs.interval`
act on scalar and vector (time series) outputs alike.
:meth:`~dualmesh.uq.Runs.bootstrap` gives a statistic with its percentile
bootstrap interval, obtained by resampling the runs, which tells whether the
number of runs suffices.

Wilks tolerance limits
~~~~~~~~~~~~~~~~~~~~~~

Safety analyses need a statement such as "with 95 % confidence, 95 % of the
possible outcomes lie below this value", whatever the distribution of the
output.  Wilks [Wilks1941]_ gave the distribution of the fraction of the
population that lies between the :math:`r`-th smallest and the :math:`r`-th
largest of :math:`n` independent runs (his Eq. (1)) and beyond one of them
(his Eq. (6)).  The :math:`r`-th largest run is therefore an upper limit for
a fraction :math:`\gamma` with the confidence

.. math:: \beta = \sum_{j=0}^{n-r} \binom{n}{j}\gamma^j(1-\gamma)^{n-j},

and the interval between the :math:`r`-th smallest and the :math:`r`-th
largest covers :math:`\gamma` with the same sum ending at :math:`n - 2r`.  For
:math:`r = 1` and :math:`\gamma = \beta = 0.95` this gives 59 runs for a
one-sided limit and 93 for a two-sided one
(:func:`~dualmesh.uq.wilks_samples`).
:meth:`~dualmesh.uq.Runs.tolerance_limit` uses the highest order :math:`r`
that meets the confidence, which gives the tightest limit the runs allow.

Correlation sensitivity
~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.uq.Runs.sensitivity` gives the correlation of each input
with an output from the same runs: the Pearson coefficient (linear effect),
the Spearman coefficient (the Pearson coefficient of the ranks, for monotonic
effects) or the partial rank correlation coefficient, which removes from the
ranks of the input and of the output the linear effect of the ranks of the
other inputs before correlating them (Marino et al. [Marino2008]_).  These
cost no extra runs and rank the inputs of monotonic models well.

Variance-based sensitivity: Sobol' indices
------------------------------------------

The output variance :math:`V(Y)` splits into the parts due to each input and
to their interactions.  The first-order index :math:`S_i = V[E(Y|X_i)]/V(Y)`
is the fraction of the variance that input :math:`i` explains on its own.
The total index :math:`S_{Ti} = E[V(Y|X_{\sim i})]/V(Y)` adds all its
interactions (Sobol' [Sobol2001]_, Definition 3).  An input with a small total
index can be fixed at its nominal value without changing the output
distribution.

:func:`~dualmesh.uq.sobol` draws two independent matrices :math:`A` and
:math:`B` of :math:`n` rows from a Sobol' sequence, and forms the :math:`k`
matrices :math:`A_B^{(i)}`, equal to :math:`A` except for column :math:`i`,
which is taken from :math:`B`.  The estimators are (b) and (f) of Table 2 of
Saltelli et al. [Saltelli2010]_, the second being that of Jansen
[Jansen1999]_, Sect. 2.3:

.. math::

    S_i \approx \frac{1}{V}\frac{1}{n}\sum_{j=1}^n f(B)_j\left[f(A_B^{(i)})_j
    - f(A)_j\right], \qquad
    S_{Ti} \approx \frac{1}{2V}\frac{1}{n}\sum_{j=1}^n\left[f(A)_j -
    f(A_B^{(i)})_j\right]^2.

Both indices of all :math:`k` inputs cost :math:`n(k + 2)` runs.  With
``second_order=True`` the matrices :math:`B_A^{(i)}` are added, and the closed
index :math:`V^c_{ij}`, the mean of :math:`f(B_A^{(i)}) f(A_B^{(j)})` less
:math:`E^2(Y)`, gives :math:`S_{ij} = (V^c_{ij} - V_i - V_j)/V(Y)` (Saltelli
[Saltelli2002]_, Theorem 2), for :math:`n(2k + 2)` runs.  The 95 % intervals
come from bootstrap resampling of the :math:`n` rows.

For an expensive model, ``surrogate="gaussian_process"`` runs the model at a
Latin hypercube of ``training_samples`` points (by default :math:`10k`, at
least 50), or takes the ``training`` runs of :func:`propagate` that are
given, fits the Gaussian process below, and computes the indices on the
process.  Only one of ``training_samples`` and ``training`` is given.  The
scrambled Sobol' design of :func:`propagate` (``method="sobol"``) is nested:
the first :math:`m` points of a design of :math:`2m` points are the design of
:math:`m` points, so that with a ``store`` a training set is enlarged by
running only the new points.  The intervals then include the uncertainty of the surrogate
itself: the indices are recomputed on random realisations of the process
(Marrel et al. [Marrel2009]_).

Gaussian process surrogate
--------------------------

A Gaussian process (kriging) emulates a model from a limited number of runs
(Sacks et al. [Sacks1989]_).  The output is :math:`y(x) = f(x)^T\beta + z(x)`:
a trend :math:`f` (a constant or a linear function) plus a stationary process
of variance :math:`\sigma^2` and correlation :math:`R(x, x')`.  The
correlation is the Matern function with smoothness 5/2 (default), 3/2, or the
squared exponential, with one length scale per input.  For Matern 5/2 and the
scaled distance :math:`r = \sqrt{\sum_p (x_p - x_p')^2/\ell_p^2}`,

.. math:: R(r) = \left(1 + \sqrt5\, r + \tfrac53 r^2\right) e^{-\sqrt5\, r}.

With the correlation matrix :math:`\mathbf R` of the training points, a nugget
:math:`\tau\mathbf I` for numerical noise, the generalised least-squares trend
:math:`\hat\beta = (F^T R^{-1}F)^{-1}F^T R^{-1}y` and the maximum-likelihood
variance :math:`\hat\sigma^2 = (y - F\hat\beta)^T R^{-1}(y - F\hat\beta)/n`,
the prediction and its variance at a point :math:`x` are (Sacks et al., Eqs.
(7) and (8))

.. math::

    \hat y(x) = f(x)^T\hat\beta + r(x)^T R^{-1}(y - F\hat\beta), \qquad
    s^2(x) = \hat\sigma^2\left[1 - r^T R^{-1} r + u^T(F^T R^{-1}F)^{-1}u\right],

with :math:`u = F^T R^{-1} r - f(x)`.  The length scales and the nugget
maximise the concentrated likelihood :math:`-\tfrac n2 \ln\hat\sigma^2 -
\tfrac12\ln|R|`, found by L-BFGS-B with the exact gradient from several
starting points.  The inputs are scaled to the unit cube.  Inputs with a
log-normal or log-uniform distribution enter by their logarithm, since a
parameter uncertain by a factor of ten usually acts through its logarithm.

A time series is standardised point by point and reduced to its principal
components, which keep 99.9 % of the variance of the training runs (the basis
representation of Higdon et al. [Higdon2008]_).  Each component score has its
own process.  The variance of the discarded components is added to the
predictive variance, so that no part of the output is predicted with more
confidence than the runs support.

:meth:`~dualmesh.uq.GaussianProcess.leave_one_out` checks the surrogate
without further runs.  With the hyperparameters fixed, the residual of
predicting point :math:`i` from the others is :math:`(Qy)_i/Q_{ii}` with
variance :math:`\hat\sigma^2/Q_{ii}`, where :math:`Q = R^{-1} -
R^{-1}F(F^TR^{-1}F)^{-1}F^TR^{-1}` (Dubrule [Dubrule1983]_, Eqs. (7) and
(8)).  From these come the predictivity coefficient :math:`Q^2 = 1 -
\sum_i(y_i - \hat y_{-i})^2/\sum_i(y_i - \bar y)^2` and standardised
residuals, which should lie within :math:`\pm 3`.  Marrel et al. regard a
surrogate with :math:`Q^2` below 0.7 as a poor approximation.

Bayesian calibration
--------------------

Calibration (inverse uncertainty quantification) finds the distribution of
the parameters :math:`\theta` that is consistent with measurements
:math:`y_E`.  The measurements and the model are related by the model
updating equation of Kennedy and O'Hagan [KennedyOHagan2001]_ and Wu et al.
[Wu2018]_,

.. math:: y_E = y_M(\theta) + \delta + \varepsilon,

where :math:`\varepsilon \sim N(0, \Sigma_\text{noise})` is the measurement
error and :math:`\delta` the model discrepancy.  By Bayes' theorem the
posterior density is

.. math::

    p(\theta|y_E) \propto p(\theta)\,|\Sigma|^{-1/2}
    \exp\left[-\tfrac12 r^T\Sigma^{-1}r\right], \qquad
    r = y_E - y_M(\theta), \qquad
    \Sigma = \Sigma_\text{noise} + \Sigma_\text{surrogate}(\theta) +
    \Sigma_\delta.

:func:`~dualmesh.uq.calibrate` puts all the outputs and all the measurements
(for example the release and the centre temperature of two rods) into this
one likelihood, in the space of the measurements.  The measurement
uncertainty is given per output as a standard deviation (a number or an
array) or as a covariance matrix.

With the default ``surrogate="gaussian_process"``, :math:`y_M` is the
surrogate fitted to ``training_samples`` runs over the priors (default
:math:`10k`, at least 30), and :math:`\Sigma_\text{surrogate}` its predictive
covariance.  In the principal components this covariance has the form
:math:`D + A\,\mathrm{diag}(v)\,A^T` with few columns in :math:`A`, and the
likelihood is evaluated by the Woodbury identity at a cost proportional to
the number of components.  With ``surrogate=None`` the model runs at every
proposal, which suits fast models only.

Without a discrepancy, :math:`\Sigma_\delta = 0`.  With
``discrepancy="gaussian_process"`` the discrepancy of each output is a
zero-mean Gaussian process in the locations of the measurements (for example
the time or the burnup, given by ``locations``), with a Matern 5/2
correlation.  It is integrated out of the likelihood, as Kennedy and O'Hagan
do (Sect. 4.4): :math:`\Sigma_\delta` is its prior covariance.  Its variance
and length scale are fixed at the joint posterior mode of the parameters and
the hyperparameters (their Sect. 4.5).  :meth:`~dualmesh.uq.Posterior.predict`
adds to the model the posterior mean of the discrepancy given the
measurements (their Eq. (10)).

The parameters and the discrepancy are confounded: several combinations of
parameter values and discrepancy fit the data equally well, so the
discrepancy widens the posterior of the parameters (Wu, Shirvan and
Kozlowski [Wu2019]_).  A discrepancy that is fitted to the residuals at fixed
parameter values and then subtracted from the data holds the posterior at
those values whatever the data, as Wu et al. [Wu2018]_ show for this case
(Sect. 5).  It is therefore not used.

Sampling and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

The posterior is sampled by four independent ensembles of the
affine-invariant stretch move of Goodman and Weare [GoodmanWeare2010]_ (their
Eqs. (7) and (9)), which needs no tuning of a proposal distribution, or by
four adaptive Metropolis chains with global adaptive scaling (Andrieu and
Thoms [AndrieuThoms2008]_, Algorithm 4).  The two halves of each ensemble move
in turn, each walker with a partner drawn from the other half.  The first
half of each chain is discarded.  :class:`~dualmesh.uq.Posterior` reports for
each parameter:

* the prior and posterior mean and standard deviation, the sample of highest
  posterior density, and the 95 % highest posterior density interval (the
  shortest interval that holds 95 % of the samples).
* the **contraction** :math:`1 - \sigma^2_\text{post}/\sigma^2_\text{prior}`.
  Near 1 the data determine the parameter.  Near 0 the posterior is the prior
  and the data carry no information about the parameter.  Wu, Shirvan and
  Kozlowski relate this lack of identifiability to a low sensitivity of the
  measured outputs to the parameter.
* an **at-bound** flag, raised when more than a quarter of the posterior lies
  in the outer 2.5 % of the prior on one side.  A parameter pushed against
  its prior often compensates for a missing physical process.
* :math:`\hat R`, the larger of the split :math:`\hat R` of the rank
  normalised draws and that of the rank normalised folded draws
  :math:`|\theta - \mathrm{median}(\theta)|` (Vehtari et al.
  [Vehtari2021]_, Eqs. (1) to (4), (14) and (15)), computed between the
  halves of the independent ensembles.  It is below 1.01 for converged
  chains, and on a surrogate the ensembles are extended until it is.
* the effective sample size, the number of samples divided by the integrated
  autocorrelation time :math:`\tau = \sum_t C(t)/C(0)` (Goodman and Weare,
  Eq. (17)), summed up to the self-consistent window that they use.

:meth:`~dualmesh.uq.Posterior.predict` gives the posterior predictive band of
every output and the fraction of the measurements it covers.

Model factors of a fuel rod
---------------------------

:class:`~dualmesh.fuel.ModelFactors` is an input group of
:class:`~dualmesh.fuel.FuelRod` (``factors=``) whose fields multiply the
models of the rod.  Every factor is 1 by default, which is the model as
published.  The ranges in the table are those used in the literature for LWR
UO2 fuel, stated as 95 % intervals.

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - Factor
     - Multiplies
     - Range in the literature
   * - ``linear_heat_rate``
     - the power history, at the same times
     - :math:`\pm 5\,\%` (Ikonen and Tulkki [Ikonen2014]_, Table 2)
   * - ``fuel_thermal_conductivity``
     - the conductivity of the fuel
     - :math:`\pm 10\,\%` (Ikonen and Tulkki, Table 2)
   * - ``cladding_thermal_conductivity``
     - the conductivity of the cladding and its coating
     - :math:`\pm 5\ \mathrm{W/(m\,K)}` (Ikonen and Tulkki, Table 2)
   * - ``gap_gas_conductance``, ``gap_contact_conductance``
     - the gas conduction and solid contact terms of the gap conductance
     - :math:`\pm 50\,\%` (Che et al. [Che2018]_)
   * - ``coolant_heat_transfer``
     - the film coefficient of forced convection
     - :math:`\pm 5\,\%` (Ikonen and Tulkki, Table 2)
   * - ``fuel_thermal_expansion``
     - the thermal expansion strain of the fuel
     - :math:`\pm 15\,\%` (Ikonen and Tulkki, Table 2)
   * - ``relocation``
     - the relocation strain of the fuel and the crack volume it opens in the
       gas balance
     -
   * - ``densification``
     - the total densification
     - :math:`\pm 20\,\%` (Che et al.)
   * - ``solid_swelling``, ``gaseous_swelling``
     - the solid and gaseous swelling strains
     -
   * - ``fuel_creep``, ``cladding_creep``
     - the creep rates
     -
   * - ``fission_gas_temperature``
     - the temperature seen by the fission gas model only
     - :math:`\pm 5\,\%` (Pastore et al. [Pastore2015]_)
   * - ``grain_radius``
     - the grain radius of the fission gas model
     - :math:`\pm 60\,\%` (Pastore et al.)
   * - ``intragranular_diffusivity``
     - the single-atom diffusion coefficient in the grains
     - a factor of 10 each way (Pastore et al.)
   * - ``resolution``
     - the re-solution rate from the intragranular bubbles
     - a factor of 10 each way (Pastore et al.)
   * - ``grain_boundary_diffusivity``
     - the vacancy diffusivity that grows the grain-face bubbles
     - a factor of 10 each way (Pastore et al.)

A factor that cannot act on the rod (a densification factor with
densification switched off, a gas factor without the Booth model) is an
error, so that a sensitivity study never reports a zero effect for a factor
that was not applied.  The property factors are applied by the
``property_scaling`` property object, which multiplies properties computed by the
property objects before it and serves any problem in the same way: every property
object also takes ``scaled_properties`` and ``property_factors`` of its own.

Verification
------------

The test suite (``tests/python/test_uq.py`` and
``tests/python/test_fuel_factors.py``) checks every part against a closed
form.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Part
     - Check and result
   * - Distributions
     - Mean, standard deviation and normalisation of all four distributions,
       truncated and untruncated, against numerical integration of the
       density (relative error below :math:`10^{-5}`).  A normal truncated to
       :math:`[6\sigma, 7\sigma]` matches the closed form to
       :math:`10^{-10}`.
   * - Propagation
     - :math:`y = 2a + 3b`: the mean 3.5 within 0.005 and the variance 0.79
       within 2 % from 4000 runs.  Over 20 designs of 256 runs, the error of
       the mean of :math:`e^{x_1 + x_2}` falls by more than half from Monte
       Carlo to Latin hypercube, and again from Latin hypercube to the Sobol'
       sequence.
   * - Wilks
     - 59, 93, 93 (second order) and 90 (95/99) runs.  In 4000 studies of
       59 normal runs the largest run exceeds the 95 % quantile in more than
       94 % of the studies (theory 95.15 %).
   * - Sobol' indices
     - The function :math:`\sin x_1 + 7\sin^2 x_2 + 0.1 x_3^4 \sin x_1` with
       :math:`n = 2048`: :math:`S_1 = 0.3145` (exact 0.3139), :math:`S_2 =
       0.4391` (0.4424), :math:`S_3 = 0.012` (0), :math:`S_{T1} = 0.5543`
       (0.5576), :math:`S_{T2} = 0.4379` (0.4424), :math:`S_{T3} = 0.2445`
       (0.2437), :math:`S_{13} = 0.238` (0.2437).  Every exact value lies in
       its bootstrap interval.  On a Gaussian process of 200 runs the indices
       are within 0.04 of the exact values.
   * - Gaussian process
     - Interpolation of the training points, :math:`Q^2 > 0.99` on 400 test
       points of the Branin function from 60 runs, more than 95 % of the
       standardised errors within :math:`\pm 3`, closed-form leave-one-out
       equal to explicit refits to :math:`10^{-6}`, and time series outputs
       through principal components.
   * - Calibration
     - A quadratic model with normal priors and noise, whose posterior is
       normal and known in closed form: posterior means within a tenth of a
       posterior standard deviation and standard deviations within 10 %
       (for example 1.2059 against 1.2036 and 0.0377 against 0.0376) for the
       ensemble and Metropolis samplers, run on the model and on a Gaussian
       process, with :math:`\hat R < 1.01`.  A parameter the data do not
       inform shows a contraction near 0, and data beyond the prior push a
       parameter to the at-bound flag.  For a fitted slope with a missing
       :math:`0.3t^2` term, the discrepancy widens the posterior more than
       fivefold and makes it independent of the prior mean.
   * - :math:`\hat R`
     - Chains with the same location and different scales: the classic split
       :math:`\hat R` is 1.00, the rank normalised and folded one 1.15.
   * - Fuel factors
     - A conductivity factor of 1.1 changes the conductivity integral
       :math:`\int_{T_s}^{T_c} k\,dT` by the factor 1.1 to within 0.5 %.  A
       power factor of 1.05 scales the burnup by 1.05 exactly, a film
       coefficient factor of 2 halves the film temperature rise, and the gas
       model with a diffusivity or temperature factor equals the model given
       the scaled coefficient or temperature exactly.
