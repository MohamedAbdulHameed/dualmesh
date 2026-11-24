Solid mechanics
===============

This page states the equations of the ``solid_mechanics`` physics: the balance
of linear momentum of a solid, its kinematics at small and finite strain, the
formulations of a body, the material laws, the conditions between bodies, the
dynamics and the structural models of beams, plates and shells.  A stress
component is positive in tension.  A displacement, traction or force component
is positive along its own coordinate axis.

.. contents::
   :local:
   :depth: 1

Balance of linear momentum
--------------------------

A solid occupies the domain :math:`\Omega`.  Its displacement
:math:`\mathbf{u}` (m) satisfies the balance of linear momentum

.. math::
   :label: solid-momentum

   \rho \frac{\partial^2 \mathbf{u}}{\partial t^2}
   = \nabla\cdot\boldsymbol{\sigma} + \mathbf{f}
   \qquad \text{in } \Omega ,

with :math:`\rho` the density (:math:`\mathrm{kg/m^3}`),
:math:`\boldsymbol{\sigma}` the Cauchy stress (Pa) and :math:`\mathbf{f}` the
body force per unit volume (:math:`\mathrm{N/m^3}`), such as the weight
:math:`\rho\mathbf{g}`.  Without inertia the left side is zero and
:eq:`solid-momentum` is the equilibrium of the body.  The balance of angular
momentum makes :math:`\boldsymbol{\sigma}` symmetric, so it has six independent
components.

Each displacement component :math:`u_i` has one equation of the balance-law
form :math:`\partial m/\partial t + \nabla\cdot\mathbf{F} = s` of
:doc:`foundations`, with

.. math::

   m = \rho \frac{\partial u_i}{\partial t} , \qquad
   \mathbf{F} = -(\sigma_{i1},\ \sigma_{i2},\ \sigma_{i3}) , \qquad
   s = f_i .

The flux :math:`\mathbf{F}` is the flux of momentum, minus row :math:`i` of the
stress.  On the boundary :math:`\partial\Omega` its outward normal component
gives the traction component
:math:`t_i = -\mathbf{F}\cdot\mathbf{n} = \sigma_{ij} n_j` (Pa), with
:math:`\mathbf{n}` the outward unit normal and repeated indices summed.  The
traction is therefore the natural boundary quantity of the displacement.

The code splits each equation between two objects.  The property object
``small_strain_stress`` (or ``finite_strain_stress``) computes the strain and
the stress at every integration point from the displacement gradients.  The
kernel ``stress_divergence`` reads the stress and assembles the flux of one
displacement.  It takes the list ``displacements`` and finds the component
:math:`i` of its equation from the place of its variable in that list.  One
kernel serves every formulation and both the finite element method (FEM) and
the dual mesh control domain method (DMCDM), because only the stress differs
between them.  The objects use the forward-mode automatic differentiation of
the code, so the Newton Jacobian holds the exact derivative of the stress with
respect to every unknown, including another field such as the temperature.

The physics ``solid_mechanics`` creates the variables ``disp_x``, ``disp_y``
and ``disp_z`` (``disp_r`` and ``disp_z`` in axisymmetric coordinates), one
stress object, one ``stress_divergence`` kernel per component and, when
``body_force`` is given, one ``body_force`` kernel per component.

Small strain
------------

Strain and Voigt order
~~~~~~~~~~~~~~~~~~~~~~

When the displacements and rotations are small, the strain is the small strain
:math:`\boldsymbol{\varepsilon}` (dimensionless), the symmetric part of the
displacement gradient,

.. math::
   :label: solid-strain

   \boldsymbol{\varepsilon} = \tfrac{1}{2}
   \left( \nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf{T}} \right) ,

and the equations hold on the undeformed body.  The code stores a symmetric
tensor as six values in the Voigt order

.. math::

   (0,\ 1,\ 2,\ 3,\ 4,\ 5) \longleftrightarrow
   (xx,\ yy,\ zz,\ yz,\ xz,\ xy) .

Slot 2 holds the hoop component :math:`\theta\theta` in the axisymmetric case.
The shear slots of a strain hold the engineering shear strains, twice the tensor
components, such as
:math:`\gamma_{xy} = 2\varepsilon_{xy} = \partial u_x/\partial y + \partial u_y/\partial x`.
The shear slots of a stress hold the tensor components.  The stress object
declares the properties ``stress``, ``strain``, ``elastic_strain``,
``volumetric_strain`` (the trace of the strain, the change of volume per unit
volume) and ``von_mises_stress``, which post-processors and output read.

Isotropic elasticity
~~~~~~~~~~~~~~~~~~~~

The stress of an isotropic linear elastic material is

.. math::
   :label: solid-hooke

   \boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon}_e)\,\mathbf{I}
   + 2G\,\boldsymbol{\varepsilon}_e ,
   \qquad
   \boldsymbol{\varepsilon}_e = \boldsymbol{\varepsilon}
   - \boldsymbol{\varepsilon}_{\mathrm{eig}}
   - \boldsymbol{\varepsilon}_c - \boldsymbol{\varepsilon}_p ,

with :math:`\mathbf{I}` the identity and :math:`\boldsymbol{\varepsilon}_e` the
elastic strain.  The elastic strain is the strain less the eigenstrains
:math:`\boldsymbol{\varepsilon}_{\mathrm{eig}}` (strains that the material takes
without stress, such as a thermal strain), the creep strain
:math:`\boldsymbol{\varepsilon}_c` and the plastic strain
:math:`\boldsymbol{\varepsilon}_p`.  The shear modulus :math:`G` (Pa) and the
first Lame parameter :math:`\lambda` (Pa) follow from Young's modulus :math:`E`
(Pa) and Poisson's ratio :math:`\nu` (dimensionless):

.. math::

   G = \frac{E}{2(1+\nu)} , \qquad
   \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)} .

Poisson's ratio satisfies :math:`-1 < \nu < 1/2`, and :math:`\lambda` grows
without bound as :math:`\nu` approaches :math:`1/2`.  Young's modulus and
Poisson's ratio have no numerical default.  Each is a constant, a function or an
expression of the fields and the coordinates.  When you leave one out, the
stress object reads the property of the same name, ``youngs_modulus`` or
``poissons_ratio``, which a property object declares.  In Voigt form the
shear rows of :eq:`solid-hooke` carry :math:`G`, because the shear slots of the
strain hold engineering strains.

Formulations
------------

The parameter ``formulation`` chooses the idealisation of the body.  A
three-dimensional mesh defaults to ``three_dimensional`` and a problem in
axisymmetric coordinates to ``axisymmetric``.  A two-dimensional Cartesian
problem names its formulation, because no plane formulation is right for every
body.

Plane stress
~~~~~~~~~~~~

``plane_stress`` describes a thin plate loaded in its plane, whose free faces
make :math:`\sigma_{zz} = \sigma_{yz} = \sigma_{xz} = 0`.  The material
contracts freely through the thickness, and the in-plane law is

.. math::

   \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix}
   = \frac{E}{1-\nu^{2}}
   \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\
     0 & 0 & \tfrac{1}{2}(1-\nu) \end{bmatrix}
   \begin{Bmatrix} \varepsilon_{e,xx} \\ \varepsilon_{e,yy} \\ \gamma_{e,xy} \end{Bmatrix} .

The ``strain`` property reports the out-of-plane strain, the elastic part

.. math::

   \varepsilon_{e,zz} = -\frac{\nu}{1-\nu}\left(\varepsilon_{e,xx} + \varepsilon_{e,yy}\right)

plus the eigenstrains and inelastic strains.  With creep or plasticity the
code solves for the :math:`zz` strain that keeps :math:`\sigma_{zz} = 0` after
the inelastic return, by Newton's method at the integration point.

Plane strain
~~~~~~~~~~~~

``plane_strain`` describes a body so long in :math:`z`, or so restrained there,
that :math:`\varepsilon_{zz} = \gamma_{yz} = \gamma_{xz} = 0`, such as a dam or
a tunnel lining.  The law is :eq:`solid-hooke` with these strains zero.  The
out-of-plane stress :math:`\sigma_{zz} = \lambda(\varepsilon_{e,xx} +
\varepsilon_{e,yy} + \varepsilon_{e,zz}) + 2G\,\varepsilon_{e,zz}` is reported
and plays no part in the in-plane equations.

Generalized plane strain
~~~~~~~~~~~~~~~~~~~~~~~~

``generalized_plane_strain`` describes the section of a long body whose axial
strain :math:`\varepsilon_{zz}` is uniform and free.  The physics adds the
variable ``strain_zz``, one unknown for the whole section, held uniform by the
constraint ``uniform_field``.  Its equation states that the axial stress carries
the axial force :math:`F_z` (N), the parameter ``axial_force``:

.. math::

   \int_\Omega \sigma_{zz}\,\mathrm{d}A = F_z ,

with :math:`\Omega` the section and :math:`|\Omega|` its area
(:math:`\mathrm{m^2}`).  The kernel ``out_of_plane_stress`` assembles the
integrand, and a ``body_force`` of :math:`F_z/|\Omega|` on ``strain_zz``
assembles the right side.  The default :math:`F_z = 0` is a long body free along its axis.

Axisymmetric
~~~~~~~~~~~~

``axisymmetric`` describes a body of revolution whose loads do not depend on
the circumferential coordinate :math:`\theta` and which has no circumferential
displacement.  The mesh is a section in the :math:`(r, z)` half-plane, with
:math:`r \ge 0` the first coordinate.  The problem is created with
``coordinates="axisymmetric"``, which gives every integral the measure
:math:`2\pi r`.  The variables are the radial displacement :math:`u_r` and the
axial displacement :math:`u_z`.

A material circle of radius :math:`r` moves to the radius :math:`r + u_r`, so
the hoop strain is

.. math::
   :label: solid-hoop

   \varepsilon_{\theta\theta} = \frac{u_r}{r} .

The stiffness is that of :eq:`solid-hooke` on the normal components
:math:`rr`, :math:`zz` and :math:`\theta\theta` and the shear :math:`rz`.  The
radial equilibrium

.. math::
   :label: solid-radial

   \frac{1}{r}\frac{\partial (r\,\sigma_{rr})}{\partial r}
   + \frac{\partial \sigma_{rz}}{\partial z}
   - \frac{\sigma_{\theta\theta}}{r} + f_r = 0

has a divergence term, which the measure :math:`2\pi r` supplies through the
flux, and the hoop term :math:`-\sigma_{\theta\theta}/r`, which no flux on the
:math:`(r, z)` mesh represents.  The kernel ``stress_divergence`` adds the hoop
term as a source of the radial equation.  The axial equation has no such term.

On the axis :math:`r = 0` the ratio :eq:`solid-hoop` is indeterminate.  A
regular solution has :math:`u_r = 0` there, so the code takes the limit
:math:`\partial u_r/\partial r` of :math:`u_r/r` at an integration point on the
axis, and a zero hoop source, where the measure :math:`2\pi r` vanishes as
well.  The condition :math:`u_r = 0` on the axis is a boundary condition that you
prescribe.

At the object level, ``small_strain_stress`` also takes ``axisymmetric_1d``: a
radial line through a long body of revolution in generalized plane strain, with
the uniform axial strain given by its parameter ``axial_strain``.

Three dimensions
~~~~~~~~~~~~~~~~

``three_dimensional`` makes no assumption about the geometry.  It takes three
displacements on a three-dimensional mesh and uses :eq:`solid-hooke` in full.

Anisotropic elasticity
----------------------

An orthotropic material takes three values each of ``youngs_modulus``
(:math:`E_1, E_2, E_3`), ``poissons_ratio`` (:math:`\nu_{12}, \nu_{13},
\nu_{23}`) and ``shear_modulus`` (:math:`G_{12}, G_{13}, G_{23}`) in its
material axes 1, 2 and 3.  They form the compliance, in the Voigt order and with
engineering shear strains,

.. math::

   \mathbf{C}^{-1} = \begin{bmatrix}
   1/E_1 & -\nu_{12}/E_1 & -\nu_{13}/E_1 \\
   -\nu_{12}/E_1 & 1/E_2 & -\nu_{23}/E_2 \\
   -\nu_{13}/E_1 & -\nu_{23}/E_2 & 1/E_3
   \end{bmatrix} \oplus \operatorname{diag}\left(1/G_{23},\ 1/G_{13},\ 1/G_{12}\right) ,

whose inverse is the stiffness :math:`\mathbf{C}` (Pa), the
:math:`6 \times 6` matrix of :math:`\boldsymbol{\sigma} = \mathbf{C}\,\boldsymbol{\varepsilon}_e`.
The code refuses constants whose compliance is not positive definite, because
they describe no stable material.  A fully anisotropic material gives
:math:`\mathbf{C}` itself as ``stiffness``, in the material axes.

The parameter ``orientation`` places the material axes.  In three dimensions it
gives axes 1 and 2 as two vectors in the global axes, and axis 3 is their cross
product.  In two dimensions it gives the angle of axis 1 from :math:`x`, in
degrees, about :math:`z`.  With :math:`\mathbf{R}` the rotation whose columns are
the material axes, the Bond matrix :math:`\mathbf{T}` maps the material stress to
the global one, :math:`\boldsymbol{\sigma}_g = \mathbf{T}\boldsymbol{\sigma}`,
and the stiffness in the global axes is

.. math::

   \mathbf{C}_g = \mathbf{T}\,\mathbf{C}\,\mathbf{T}^{\mathsf{T}} .

In plane stress the code solves for the out-of-plane strains
:math:`\varepsilon_{zz}`, :math:`\gamma_{yz}` and :math:`\gamma_{xz}` that make
:math:`\sigma_{zz}`, :math:`\sigma_{yz}` and :math:`\sigma_{xz}` zero.  An
anisotropic material is available at small strain.

Thermal expansion
-----------------

A change of temperature :math:`T` (K) makes a free body expand without stress.
The coupling ``thermal_expansion`` of a ``solid_mechanics`` and a
``heat_transfer`` physics adds the thermal eigenstrain

.. math::
   :label: solid-thermal

   \boldsymbol{\varepsilon}_{\mathrm{th}} = \alpha\,(T - T_0)\,\mathbf{I} ,

with :math:`\alpha` the linear expansion coefficient
(:math:`\mathrm{1/K}`, ``linear_expansion_coefficient``) and :math:`T_0` the
temperature at which the solid is free of thermal strain (K,
``stress_free_temperature``).  The property object
``thermal_expansion_eigenstrain`` computes :eq:`solid-thermal` on the three
normal components, and the stress object subtracts it in :eq:`solid-hooke`.  An
isotropic expansion has no shear part.  :math:`T_0` has no default, because no
value is right for every problem.

A body heated above :math:`T_0` and restrained in its plane is in compression.
In plane strain, where the out-of-plane strain is zero as well, it carries
:math:`\sigma_{xx} = \sigma_{yy} = -E\alpha(T - T_0)/(1 - 2\nu)`.  In plane
stress it expands freely through its thickness and carries
:math:`\sigma_{xx} = \sigma_{yy} = -E\alpha(T - T_0)/(1 - \nu)`.

The temperature is the field of the ``heat_transfer`` physics.  When that field
is an unknown of the same problem, the stress depends on it, and the automatic
differentiation carries :math:`\partial\boldsymbol{\sigma}/\partial T` into the
Jacobian.  Newton's method then converges at its usual rate on the coupled
system.  In a beam or a plate, the section integrates a thermal strain that is
uniform through the thickness into a thermal force and a thermal moment.

Finite strain
-------------

With ``large_deformation=True``, or with a hyperelastic material, the equations
account for the change of shape.  They hold on the undeformed mesh (the total
Lagrangian form).  The deformation gradient :math:`\mathbf{F}_d` (dimensionless)
is taken with respect to the undeformed position :math:`\mathbf{X}` (m), and the
equilibrium uses the first Piola-Kirchhoff stress :math:`\mathbf{P}` (Pa):

.. math::

   \mathbf{F}_d = \mathbf{I} + \nabla_{\mathbf{X}}\mathbf{u} , \qquad
   J = \det\mathbf{F}_d , \qquad
   \mathbf{P} = J\,\boldsymbol{\sigma}\,\mathbf{F}_d^{-\mathsf{T}} , \qquad
   \nabla_{\mathbf{X}}\cdot\mathbf{P} + \mathbf{f}_0 = \mathbf{0} ,

with :math:`J` the ratio of deformed to undeformed volume and
:math:`\mathbf{f}_0` the body force per unit undeformed volume.  The property
object ``finite_strain_stress`` declares :math:`\mathbf{P}` as
``first_piola_kirchhoff_stress`` (nine components, row by row), which
``stress_divergence`` reads.  It also declares ``deformation_gradient`` and
``green_lagrange_strain``, the strain
:math:`\tfrac{1}{2}(\mathbf{F}_d^{\mathsf{T}}\mathbf{F}_d - \mathbf{I})`.
``stress`` holds the Cauchy stress and ``volumetric_strain`` holds
:math:`J - 1`.  Finite strain is available in plane strain, axisymmetric and
three-dimensional problems.

The parameter ``elasticity`` chooses the law.

- ``linear`` with ``large_deformation=True`` applies the rotated small strain of
  Hughes and Winget.  Over a step, with :math:`\Delta\mathbf{F}_d` the change of
  :math:`\mathbf{F}_d` and :math:`\mathbf{F}_{d,1/2}` its midpoint value, the
  increment gradient :math:`\mathbf{L} = \Delta\mathbf{F}_d\,\mathbf{F}_{d,1/2}^{-1}`
  gives the strain increment :math:`\operatorname{sym}\mathbf{L}` and the
  rotation increment
  :math:`\mathbf{Q} = (\mathbf{I} - \tfrac{1}{2}\mathbf{W})^{-1}(\mathbf{I} + \tfrac{1}{2}\mathbf{W})`,
  with :math:`\mathbf{W} = \operatorname{skew}\mathbf{L}`.  The elastic strain
  of the last step is rotated by :math:`\mathbf{Q}` and the strain increment is
  added:

  .. math::

     \boldsymbol{\varepsilon}_e^{n+1} = \mathbf{Q}\,\boldsymbol{\varepsilon}_e^{n}\,\mathbf{Q}^{\mathsf{T}}
     + \operatorname{sym}\mathbf{L} - \Delta\boldsymbol{\varepsilon}_{\mathrm{eig}} .

  A rotation therefore makes no stress.  The stress follows from
  :eq:`solid-hooke`, with the creep and plastic returns of small strain, and the
  back stresses of kinematic hardening rotate with :math:`\mathbf{Q}`.  This law
  suits materials whose elastic strains stay small while their total strains
  are large, such as metals and ceramics.
- ``neo-Hookean`` is the compressible neo-Hookean law

  .. math::

     \mathbf{P} = G\left(\mathbf{F}_d - \mathbf{F}_d^{-\mathsf{T}}\right)
     + \lambda \ln J\,\mathbf{F}_d^{-\mathsf{T}} ,

  exact for any deformation.  An eigenstrain enters as
  :math:`\mathbf{F}_d = \mathbf{F}_e\mathbf{F}_{\mathrm{eig}}`, with the law
  applied to the elastic part :math:`\mathbf{F}_e`.  The neo-Hookean law has no
  creep or plasticity.

A ``pressure`` at finite strain acts on the deformed surface.  Per unit
undeformed area its traction is
:math:`-p\,J\,\mathbf{F}_d^{-\mathsf{T}}\mathbf{N}` (Nanson's relation), with
:math:`p` the pressure (Pa) and :math:`\mathbf{N}` the undeformed outward
normal.

Creep and plasticity
--------------------

Creep and plasticity add a history at every integration point: the creep strain
:math:`\boldsymbol{\varepsilon}_c`, the plastic strain
:math:`\boldsymbol{\varepsilon}_p`, their equivalent values
:math:`\bar\varepsilon_c` and :math:`\bar\varepsilon_p` (dimensionless) and the
back stresses.  The code integrates them by the backward Euler method.  The
increments solve their equations at the point, and their derivatives with
respect to the unknowns follow from the implicit function theorem, so the
Jacobian stays exact.  Both laws read the temperature: a field, a number, or the
temperature of a ``thermal_expansion`` coupling.

Creep
~~~~~

The parameter ``creep`` gives the equivalent creep strain rate
:math:`\dot{\bar\varepsilon}_c` (:math:`\mathrm{1/s}`) as an expression of
``von_mises_stress`` (Pa), ``temperature`` (K), ``grain_size`` (m), the time
``t`` (s) and ``equivalent_creep_strain``, with named constants.  The creep
strain flows along the deviatoric stress
:math:`\operatorname{dev}\boldsymbol{\sigma}`, the stress less its mean.  With
:math:`\sigma_{\mathrm{eq}} = \sqrt{\tfrac{3}{2}\operatorname{dev}\boldsymbol{\sigma} :
\operatorname{dev}\boldsymbol{\sigma}}` the von Mises stress (Pa) of the trial
stress, the stress of the elastic strain before the return, the radial return
solves

.. math::

   \Delta\bar\varepsilon_c = \Delta t\,\dot{\bar\varepsilon}_c\left(
   \sigma_{\mathrm{eq}} - 3G\,\Delta\bar\varepsilon_c,\ T,\
   \bar\varepsilon_c^{\,n} + \Delta\bar\varepsilon_c \right) ,
   \qquad
   \Delta\boldsymbol{\varepsilon}_c = \tfrac{3}{2}\,\Delta\bar\varepsilon_c\,
   \frac{\operatorname{dev}\boldsymbol{\sigma}}{\sigma_{\mathrm{eq}}} ,

with :math:`\Delta t` the time step (s).  The root lies in
:math:`[0,\ \sigma_{\mathrm{eq}}/(3G)]`, so the stress cannot reverse.  Creep
acts in a time step only.

J2 plasticity
~~~~~~~~~~~~~

J2 plasticity yields on the second invariant of the deviatoric stress, through
the von Mises stress.  The parameter ``plasticity`` gives the yield stress
:math:`\sigma_y` (Pa) as an
expression of ``equivalent_plastic_strain`` and ``temperature``, which sets the
isotropic hardening.  A number is perfect plasticity, and any hardening law that
is an expression of these two, such as a Voce law, is accepted.  The yield
function is

.. math::

   \sigma_{\mathrm{eq}}(\boldsymbol{\xi}) - \sigma_y(\bar\varepsilon_p, T) \le 0 ,
   \qquad
   \boldsymbol{\xi} = \operatorname{dev}\boldsymbol{\sigma} - \sum_k \mathbf{X}_k ,

with :math:`\boldsymbol{\xi}` the relative stress and :math:`\mathbf{X}_k` the
back stresses (Pa) of kinematic hardening.  The flow is normal to the yield
surface.  Each back stress follows the law of Armstrong and Frederick,

.. math::

   \mathrm{d}\mathbf{X}_k = \tfrac{2}{3} C_k\,\mathrm{d}\boldsymbol{\varepsilon}_p
   - \gamma_k\,\mathbf{X}_k\,\mathrm{d}\bar\varepsilon_p ,

with :math:`C_k` its modulus (Pa) and :math:`\gamma_k` its dynamic recovery
(dimensionless), given as ``kinematic_hardening`` ``{"C": [...], "gamma": [...]}``.
Several back stresses make the Chaboche model, and :math:`\gamma_k = 0` makes
Prager's linear hardening.  The backward Euler step with
:math:`a_k = 1/(1 + \gamma_k\Delta\bar\varepsilon_p)` makes the relative stress
at the end of the step parallel to
:math:`\tilde{\boldsymbol{\xi}} = \operatorname{dev}\boldsymbol{\sigma}^{\mathrm{trial}}
- \sum_k a_k \mathbf{X}_k^n`, so the return is one scalar equation,

.. math::

   \sigma_{\mathrm{eq}}(\tilde{\boldsymbol{\xi}})
   - \Bigl(3G + \sum_k a_k C_k\Bigr)\Delta\bar\varepsilon_p
   - \sigma_y\left(\bar\varepsilon_p^{\,n} + \Delta\bar\varepsilon_p\right) = 0 ,

which Newton's method solves inside a bracket of the root.  Plasticity acts in
a steady solve and in a time step.

With creep and isotropic hardening together, both strains flow along the
deviator of the trial stress, whose von Mises value falls to
:math:`\sigma_{\mathrm{eq}} - 3G(\Delta\bar\varepsilon_c + \Delta\bar\varepsilon_p)`.
The two increments solve the creep equation and, where the point yields, the
yield condition at that stress, together.  Creep with kinematic hardening is
not available.

Anisotropic creep and plasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key ``hill`` of ``creep`` or ``plasticity`` gives Hill's six coefficients
:math:`\mathcal{F}, \mathcal{G}, \mathcal{H}, \mathcal{L}, \mathcal{M},
\mathcal{N}` (dimensionless) in the material axes.  They define the equivalent
stress :math:`\sigma_{\mathrm{eq}} = \sqrt{\boldsymbol{\sigma}\cdot\mathbf{Y}\boldsymbol{\sigma}}`,
with, in the Voigt order with tensor shear stresses,

.. math::

   \mathbf{Y} = \begin{bmatrix}
   \mathcal{G}+\mathcal{H} & -\mathcal{H} & -\mathcal{G} \\
   -\mathcal{H} & \mathcal{F}+\mathcal{H} & -\mathcal{F} \\
   -\mathcal{G} & -\mathcal{F} & \mathcal{F}+\mathcal{G}
   \end{bmatrix} \oplus \operatorname{diag}\left(2\mathcal{L},\ 2\mathcal{M},\ 2\mathcal{N}\right) ,

turned to the global axes by the Bond matrix.  The values
:math:`\mathcal{F} = \mathcal{G} = \mathcal{H} = 1/2` and
:math:`\mathcal{L} = \mathcal{M} = \mathcal{N} = 3/2` give the von Mises stress.
With Hill's stress or an anisotropic stiffness, the return is the closest-point
projection: Newton's method on the stress and the two increments at the point,
with

.. math::

   \mathbf{C}^{-1}\boldsymbol{\sigma} - \boldsymbol{\varepsilon}_e^{\mathrm{trial}}
   + \left(\Delta\bar\varepsilon_c + \Delta\bar\varepsilon_p\right)
   \frac{\partial\sigma_{\mathrm{eq}}}{\partial\boldsymbol{\sigma}} = \mathbf{0} .

Hill plasticity and an anisotropic stiffness are not available with kinematic
hardening.

Viscoelasticity
---------------

The parameter ``viscoelasticity`` makes a generalized Maxwell material:
branches that add to the long-term moduli :math:`E` and :math:`\nu`.  A shear
branch of modulus :math:`G_i` (Pa) and relaxation time :math:`\tau_i` (s)
carries the deviatoric stress :math:`\boldsymbol{\sigma}_i` (Pa), which relaxes
as

.. math::

   \dot{\boldsymbol{\sigma}}_i = 2G_i\,\dot{\mathbf{e}} - \frac{\boldsymbol{\sigma}_i}{\tau_i} ,

with :math:`\mathbf{e}` the deviatoric strain.  For a strain that varies
linearly over the step, the update of Herrmann and Taylor is exact:

.. math::

   \boldsymbol{\sigma}_i^{n+1} = e^{-\xi_i}\,\boldsymbol{\sigma}_i^{n}
   + 2G_i\,\frac{1 - e^{-\xi_i}}{\xi_i}\left(\mathbf{e}^{n+1} - \mathbf{e}^{n}\right) ,
   \qquad \xi_i = \frac{\Delta t}{\tau_i} .

A bulk branch of modulus :math:`K_i` (Pa) updates its mean stress from the
volumetric strain in the same way.  A steady solve gives the relaxed state, in
which the branches carry no stress.  A step of zero length gives the
instantaneous response.  Viscoelasticity is available at small strain, outside
plane stress, and without creep or plasticity.

Fracture
--------

Crack-tip integrals
~~~~~~~~~~~~~~~~~~~

For a crack along :math:`x` in plane strain or plane stress, with an isotropic
material of constant moduli,
:meth:`~dualmesh.physics.SolidMechanics.j_integral` and
:meth:`~dualmesh.physics.SolidMechanics.stress_intensity` compute the domain
integrals over the ring between an inner and an outer radius about the tip.  The
weight :math:`\zeta` (dimensionless) is 1 inside the inner radius, 0 outside the outer one and
linear between, and in the crack axes

.. math::

   \mathcal{J} = \int_{\mathrm{ring}} \left( \sigma_{ij}\,\frac{\partial u_j}{\partial x_1}
   - \psi\,\delta_{1i} \right) \frac{\partial \zeta}{\partial x_i}\,\mathrm{d}A ,

with :math:`\mathcal{J}` the J-integral (:math:`\mathrm{J/m^2}`), :math:`\psi` the
strain energy density (:math:`\mathrm{J/m^3}`) and :math:`\delta_{1i}` the
Kronecker delta.  The stress intensity factors :math:`K_I` and :math:`K_{II}`
(:math:`\mathrm{Pa\,m^{1/2}}`) follow from the interaction integral
:math:`\mathcal{I}` with the Williams fields of mode I or II at unit intensity,
:math:`K = E'\mathcal{I}/2`, with :math:`E' = E` in plane stress and
:math:`E' = E/(1-\nu^2)` in plane strain.  The integrals take the faces of the
crack free of traction.

Phase-field fracture
~~~~~~~~~~~~~~~~~~~~

The parameter ``phase_field_fracture`` adds the damage field
:math:`d` (dimensionless, 0 intact and 1 broken) of the second
Ambrosio-Tortorelli model (AT2), with the toughness :math:`G_c`
(:math:`\mathrm{J/m^2}`, ``toughness``) and the width of the diffuse crack
:math:`\ell` (m, ``length``).  The split of Miehe divides the elastic energy by
the signs of the principal strains :math:`\varepsilon_k` and of the trace,

.. math::

   \psi^+ = \tfrac{1}{2}\lambda\,\langle\operatorname{tr}\boldsymbol{\varepsilon}_e\rangle_+^2
   + G \sum_k \langle\varepsilon_k\rangle_+^2 ,

with :math:`\langle x\rangle_+ = \max(x, 0)`.  A zero strain belongs to the
compressive part, so the undeformed state has the stiffness of the intact
material.  The tensile stress is degraded by :math:`(1-d)^2` and the compressive
stress is kept.  The damage solves

.. math::

   -G_c\,\ell\,\nabla^2 d + \left(\frac{G_c}{\ell} + 2\psi^+_{\max}\right) d
   = 2\psi^+_{\max} ,
   \qquad \psi^+_{\max}(t) = \max_{t' \le t} \psi^+(t') ,

so a crack costs the energy :math:`G_c` per unit area and does not heal.
Phase-field fracture is available for an isotropic elastic material at small
strain in plane strain, generalized plane strain and axisymmetric problems.

Conditions between bodies
-------------------------

Contact and cohesive zones act between two surfaces of a two-dimensional problem
meshed with linear elements.

Contact
~~~~~~~

The condition ``contact`` with ``primary`` and ``secondary`` surfaces holds the
bodies apart by the mortar method.  The contact pressure :math:`p_{c,j}` (Pa) at
node :math:`j` of the secondary surface is a Lagrange multiplier, the variable
``contact_pressure``.  The weighted gap of the node is

.. math::

   g_j = \int N_j\,(\mathbf{x}_p - \mathbf{x}_s)\cdot\mathbf{n}\,\mathrm{d}S ,

with :math:`N_j` the shape function of the node, :math:`\mathbf{x}_s` a point of
the secondary surface, :math:`\mathbf{x}_p` the primary point it projects to
along the secondary normal :math:`\mathbf{n}`, and the integral over the
secondary surface.  The gap and the pressure satisfy
:math:`g_j \ge 0`, :math:`p_{c,j} \ge 0` and :math:`p_{c,j}\,g_j = 0`, which the
code writes as the complementarity function

.. math::

   p_{c,j} - \max\left(0,\ p_{c,j} - c\,\frac{g_j}{\bar N_j}\right) = 0 ,
   \qquad \bar N_j = \int N_j\,\mathrm{d}S ,

and solves by Newton's method with the active set.  The scale :math:`c`
(:math:`\mathrm{Pa/m}`), Young's modulus over the mean side length of the
secondary surface, changes the convergence and leaves the solution unchanged.
The pressure acts as :math:`-p_{c,j}\mathbf{n}` on the secondary surface and
:math:`+p_{c,j}\mathbf{n}` on the primary one, so the forces balance.  With a
``friction_coefficient``, Coulomb friction adds the tangential traction as a
second multiplier, with the complementarity function of Hüeber, Stadler and
Wohlmuth on the weighted slip.

Cohesive zones
~~~~~~~~~~~~~~

The condition ``cohesive`` bonds two surfaces that coincide at the start, with
the ``strength`` :math:`\sigma_c` (Pa) and the ``toughness`` :math:`G_c`
(:math:`\mathrm{J/m^2}`).  The code pairs the points of the two surfaces once,
on the undeformed surfaces.  The separation of a pair,
:math:`\boldsymbol{\delta} = \mathbf{x}_p - \mathbf{x}_s` (m), has the normal
part :math:`\delta_n` and the tangential part :math:`\delta_t`, and the traction
on the secondary surface is

.. math::

   \mathbf{t} = K_c\,(1 - D_c)\,\boldsymbol{\delta} ,

with :math:`K_c` the stiffness of the bond (:math:`\mathrm{Pa/m}`) and
:math:`D_c` its damage.  A closing normal part, :math:`\delta_n < 0`, keeps the
full stiffness, so the surfaces do not pass through each other.  The damage
grows with the largest effective opening reached,
:math:`\delta_{\max} = \max \sqrt{\langle\delta_n\rangle_+^2 + \delta_t^2}`:

.. math::

   D_c = \frac{\delta_f\,(\delta_{\max} - \delta_0)}{\delta_{\max}\,(\delta_f - \delta_0)} ,
   \qquad \delta_0 = \frac{\sigma_c}{K_c} , \qquad \delta_f = \frac{2G_c}{\sigma_c} ,

with :math:`D_c = 0` below :math:`\delta_0` and :math:`D_c = 1` above
:math:`\delta_f`.  The traction peaks at :math:`\sigma_c`.  The dissipation
:math:`\tfrac{1}{2}K_c\,\delta_{\max}^2\,\mathrm{d}D_c` integrates from
:math:`\delta_0` to :math:`\delta_f` to
:math:`\tfrac{1}{2}\sigma_c\,\delta_f = G_c` for any mix of opening and
sliding.  The physics sets :math:`K_c = 50\,E/h`, with :math:`h` the mean side
length of the secondary surface.

Dynamics
--------

A ``density`` adds the inertia :math:`\rho\,\partial^2\mathbf{u}/\partial t^2`
of :eq:`solid-momentum`.  A transient solve integrates it by the generalized-alpha
method of Chung and Hulbert [ChungHulbert1993]_, or by the average acceleration
rule of Newmark, the ``time_scheme`` ``"Newmark"`` of
:meth:`~dualmesh.Problem.solve_transient`.  The parameter ``spectral_radius``
:math:`\rho_\infty` of the generalized-alpha method sets the damping of the
modes that the mesh cannot resolve, from 0 to 1, with the default 0.75.
:doc:`nonlinear_and_time` states both schemes.  An eigenvalue study takes the
density for the natural frequencies.  The parameter ``initial_condition`` gives
the initial ``displacement`` and ``velocity``, and the default is a body
undeformed and at rest.

The parameter ``damping`` adds Rayleigh damping
:math:`a\,\mathbf{M} + b\,\mathbf{K}`, with :math:`\mathbf{M}` the mass matrix,
:math:`\mathbf{K}` the elastic stiffness at small strain, :math:`a`
(:math:`\mathrm{1/s}`, ``mass``) and :math:`b` (s, ``stiffness``).  The mass
part is a first time derivative with the coefficient :math:`a\rho`.  The
stiffness part is the kernel ``stiffness_damping``, the flux of the stress rate
:math:`b\,\mathbf{C}\,\dot{\boldsymbol{\varepsilon}}` of the velocity of the
step.

Boundary conditions
-------------------

At every boundary point each displacement component has either its value or its
traction component prescribed, independently of the other components.  A
boundary with no condition for a component has zero traction, so a free surface
needs no input.

- ``Dirichlet`` prescribes :math:`u_i` at the nodes of a boundary.  Zero is a
  support in that direction, and all components zero is a clamp.
- ``symmetry`` holds the displacement normal to a plane of symmetry at zero and
  leaves the tangential ones free.  The side set lies in a plane normal to a
  coordinate axis.
- ``normal_displacement`` prescribes :math:`\mathbf{u}\cdot\mathbf{n}` on an
  inclined or curved face and leaves the tangential displacements free.  The
  normal force at each node is a Lagrange multiplier, and the normal of a node
  is the mean of the normals of its sides, weighted by their areas.
- ``rotation`` turns a boundary by an ``angle`` (rad) about an axis through a
  ``center``.  At small strain it prescribes
  :math:`\mathbf{u} = \vartheta\,\mathbf{k}\times(\mathbf{x} - \mathbf{c})`, with
  :math:`\vartheta` the angle, :math:`\mathbf{k}` the unit axis and
  :math:`\mathbf{c}` the center.  At finite strain it prescribes the rotation of
  Rodrigues.
- ``traction`` prescribes the traction component of its variable, in force per
  unit area.  A positive value acts along the positive axis of that variable
  whichever way the outward normal points.  ``total_force`` gives the total
  force instead, spread uniformly over the area of the boundary.
- ``pressure`` gives the traction :math:`t_i = -p\,n_i`.  A positive pressure
  presses inward, and a negative one is a suction.
- ``torque`` applies a torque :math:`M_t` (N m) as the traction
  :math:`M_t\,\mathbf{k}\times\mathbf{r}/J_p` of the torsion of a circular shaft,
  with :math:`\mathbf{r} = \mathbf{x} - \mathbf{c}` and
  :math:`J_p = \int |\mathbf{k}\times\mathbf{r}|^2\,\mathrm{d}A` the polar moment
  of the boundary (:math:`\mathrm{m^4}`).
- ``point_source`` applies a concentrated force at the nodes of a boundary or at
  the node nearest to a point.  The continuum stress under a point force is
  unbounded, so the stress next to it depends on the mesh, while the
  displacements away from it converge.

``rotation`` and ``torque`` act in plane strain, plane stress and
three-dimensional problems.

Reactions
~~~~~~~~~

At a node with a prescribed displacement the code replaces the equation of that
component and keeps its residual.  At the converged solution the residual is
the force that the support transmits to the body at the node.
:meth:`~dualmesh.physics.SolidMechanics.reaction_force` sums it over a
boundary.  The reaction of a node is that of its whole equation, so a corner
node of two constrained boundaries counts in each of them.  To sum the reaction
over several boundaries, merge their nodes into one boundary of the mesh.

The thickness of a plane problem
--------------------------------

The parameter ``thickness`` :math:`H` (m) is the out-of-plane size of a plane
problem: the thickness of the sheet in plane stress, or the length of the slice
in plane strain.  It multiplies the flux of ``stress_divergence``, the body
force, the ``traction``, the ``pressure`` and the contact conditions, so every
residual is a force.  ``traction`` and ``pressure`` read :math:`H` from the
``stress_divergence`` kernel of their displacement, and the physics passes the
same :math:`H` to the contact conditions, so the loads and the stiffness scale
together.  A condition takes one thickness, so it refuses blocks of different
thickness.  The default is 1 m.  A three-dimensional,
axisymmetric or generalized plane strain problem takes no thickness, because the
measure :math:`2\pi r` or the section already gives its extent.

A ``point_source`` is a total force and has no thickness factor.  A line load
:math:`w_z` (:math:`\mathrm{N/m}`) along the out-of-plane direction is therefore
the point force :math:`w_z H`.

Structural models
-----------------

A beam, a plate or a shell is a three-dimensional elastic body whose
displacement through the thickness is restricted, and whose equations are
integrated through it.  The code writes every model as a system of
second-order equations in the balance-law form, which DMCDM requires.  The
models are linear elastic at small strain, with the option ``von_karman`` of
moderately large deflections for beams and plates.  The theories of beams and
plates are those of [ReddyBeams2022]_ and [ReddyPlates2007]_, and their dual
mesh forms those of [Reddy2024]_.

Sections
~~~~~~~~

A section is given by its material, ``youngs_modulus`` and ``poissons_ratio``
with the ``thickness`` :math:`H` of a plate or the ``cross_section`` of a beam,
or by its section stiffnesses for a laminated or graded section:

- :math:`A`, ``extensional_stiffness`` (N), the integral of :math:`E` over the
  section,
- :math:`B`, ``coupling_stiffness`` (N m), the integral of :math:`E\,z`, zero for
  a section symmetric about its mid-surface,
- :math:`D`, ``bending_stiffness`` (:math:`\mathrm{N\,m^2}`), the integral of
  :math:`E\,z^2`,
- :math:`S`, ``shear_stiffness`` (N), the transverse shear stiffness
  :math:`K_s G A`.

Here :math:`z` (m) is the position through the thickness from the mid-surface,
and :math:`K_s` (dimensionless) is the shear correction factor, 5/6 by default.
Plate stiffnesses are per unit width and carry the factor
:math:`1/(1-\nu^2)`.  A section given by its material integrates the law at
Gauss points through the thickness, so :math:`E(z)` gives a functionally graded
section.  A beam section is ``{"width": ..., "height": ...}``, a rectangle, or
``{"area": ..., "second_moment": ...}``, and a Timoshenko beam of the second
form needs its ``shear_correction``.

The shear terms of the models with shear deformation are integrated at the
centre of each element, which keeps thin beams and plates from locking.  The
loads are the ``transverse_load`` :math:`q` (per unit length of a beam or unit
area of a plate), the ``axial_load`` :math:`\bar f` of a beam (the radial load of a
circular plate), the ``in_plane_load`` :math:`(\bar f_x, \bar f_y)` of a plate, and the
Winkler ``foundation_modulus`` :math:`c_f`.

Beams
~~~~~

``formulation="beam"`` takes a one-dimensional mesh of ``Edge2`` elements along
:math:`x`, the axial displacement :math:`u` and the deflection :math:`w`.  With
the axial strain :math:`\varepsilon_0 = u' + \tfrac{1}{2}w'^2`, where the prime
is :math:`\mathrm{d}/\mathrm{d}x` and the square term belongs to the von Kármán
option, the theories are:

- Euler-Bernoulli (the default) and Timoshenko in the mixed form, with the
  bending moment :math:`M` (N m) as the third variable, ``bending_moment``:

  .. math::

     -\left(\bar A\,\varepsilon_0 + \bar B M\right)' = \bar f , \qquad
     -\left(M' + N_x w'\right)' + c_f w = q , \qquad
     -\left(w' - \frac{M'}{S}\right)' + \bar B\,\varepsilon_0 - \frac{M}{D} = 0 ,

  with :math:`\bar A = (DA - B^2)/D`, :math:`\bar B = B/D` and
  :math:`N_x = \bar A\,\varepsilon_0 + \bar B M` the axial force (N).  The
  Euler-Bernoulli beam has no shear flexibility, :math:`1/S = 0`.  The natural
  condition of the third equation is a zero slope, so the physics holds the
  moment at zero on every end that is not clamped, not a plane of symmetry and
  has no moment of its own.
- Timoshenko in the displacement form, ``beam_formulation="displacement"``,
  with the rotation :math:`\phi` (rad) as the third variable, ``phi``:

  .. math::

     -\left(A\varepsilon_0 + B\phi'\right)' = \bar f , \qquad
     -\left(S(\phi + w') + N_x w'\right)' + c_f w = q , \qquad
     -\left(B\varepsilon_0 + D\phi'\right)' + S(\phi + w') = 0 ,

  with :math:`N_x = A\varepsilon_0 + B\phi'`.

Plates
~~~~~~

``formulation="plate"`` is the first-order shear deformation (Mindlin) theory on
a mesh of ``Quad4`` elements of the mid-plane, with the variables ``disp_x``,
``disp_y``, ``disp_z`` (:math:`u, v, w`) and the rotations ``phi_x``, ``phi_y``
(:math:`\phi_x, \phi_y`).  The membrane strains are
:math:`\varepsilon_{xx} = \partial u/\partial x + \tfrac{1}{2}(\partial w/\partial x)^2`,
:math:`\varepsilon_{yy}` likewise and
:math:`\gamma_{xy} = \partial u/\partial y + \partial v/\partial x + (\partial w/\partial x)(\partial w/\partial y)`,
whose terms in :math:`w` belong to the von Kármán option, and the curvatures are :math:`\chi_{xx} = \partial\phi_x/\partial x`,
:math:`\chi_{yy} = \partial\phi_y/\partial y` and
:math:`\chi_{xy} = \partial\phi_x/\partial y + \partial\phi_y/\partial x`.  The
resultants are

.. math::

   N_{xx} = A(\varepsilon_{xx} + \nu\varepsilon_{yy}) + B(\chi_{xx} + \nu\chi_{yy}) , \qquad
   N_{xy} = \tfrac{1}{2}(1-\nu)(A\gamma_{xy} + B\chi_{xy}) ,

.. math::

   M_{xx} = B(\varepsilon_{xx} + \nu\varepsilon_{yy}) + D(\chi_{xx} + \nu\chi_{yy}) , \qquad
   M_{xy} = \tfrac{1}{2}(1-\nu)(B\gamma_{xy} + D\chi_{xy}) , \qquad
   Q_x = S\left(\phi_x + \frac{\partial w}{\partial x}\right) ,

with the :math:`y` components alike.  With a thermal strain, the thermal force
and moment are subtracted from the normal resultants.  The equations are

.. math::

   \frac{\partial N_{xa}}{\partial x_a} + \bar f_x = 0 , \qquad
   \frac{\partial}{\partial x_a}\left(Q_a + N_{ab}\frac{\partial w}{\partial x_b}\right)
   - c_f w + q = 0 , \qquad
   \frac{\partial M_{xa}}{\partial x_a} - Q_x = 0 ,

with the indices :math:`a, b` summed over :math:`x` and :math:`y`, the term in
:math:`N_{ab}` that of the von Kármán option, and the :math:`y` equations
alike.

``formulation="circular_plate"`` is an axisymmetric plate on a radial mesh in
axisymmetric coordinates, with the variables ``disp_r``, ``disp_z`` and the
rotation ``phi`` (first-order theory, the default) or the radial moment
``bending_moment`` (classical theory, in the mixed form of [Reddy2024]_, which
turns the fourth-order equation of the deflection into three second-order
equations).

Trusses
~~~~~~~

``formulation="truss"`` takes a line mesh of ``Edge2`` bars in a space of one,
two or three dimensions, with the area :math:`A_b` (:math:`\mathrm{m^2}`) as
``cross_section={"area": ...}``.  A bar of unit direction :math:`\mathbf{e}`
carries the axial force :math:`E A_b\varepsilon`, with
the stretch :math:`\varepsilon = \mathbf{e}\cdot\partial\mathbf{u}/\partial s`
and :math:`s` (m) the length along the bar, and nothing across it.  The equation
of :math:`u_i` has the flux :math:`-E A_b\varepsilon\,e_i\,\mathbf{e}`.  A bar has
no stiffness across it, so a bar meshed with more than one element has free
interior nodes, and each bar takes one element.

Shells
~~~~~~

``formulation="shell"`` takes a surface mesh of ``Quad4`` elements in three
dimensions, with the ``thickness`` :math:`H`.  Each element is a flat shell with
its own axes :math:`\mathbf{e}_1, \mathbf{e}_2` in its plane and
:math:`\mathbf{e}_3` normal to it.  The unknowns are the displacements
``disp_x``, ``disp_y``, ``disp_z`` and the rotations ``rotation_x``,
``rotation_y``, ``rotation_z`` about the global axes, :math:`\boldsymbol{\vartheta}`
(rad).  In the axes of the element they are
:math:`\mathbf{u}_l = \mathbf{R}_e\mathbf{u}` and
:math:`\boldsymbol{\vartheta}_l = \mathbf{R}_e\boldsymbol{\vartheta}`, with
:math:`\mathbf{R}_e` the matrix of rows :math:`\mathbf{e}_1, \mathbf{e}_2,
\mathbf{e}_3`, and the rotations of the normal are
:math:`\boldsymbol{\phi} = (\vartheta_{l2},\ -\vartheta_{l1})`.  With the
indices :math:`a, b` over the two in-plane axes:

.. math::

   \varepsilon_{ab} = \operatorname{sym}\partial_a u_{l,b} , \qquad
   \chi_{ab} = \operatorname{sym}\partial_a \phi_b , \qquad
   \gamma_a = \partial_a w_l + \phi_a , \qquad
   \omega = \vartheta_{l3} - \tfrac{1}{2}\left(\partial_1 u_{l,2} - \partial_2 u_{l,1}\right) ,

with :math:`w_l` the displacement along :math:`\mathbf{e}_3`.  The resultants are
:math:`\mathbf{N} = A_m[(1-\nu)\boldsymbol{\varepsilon} + \nu\operatorname{tr}\boldsymbol{\varepsilon}\,\mathbf{I}]`,
:math:`\mathbf{M} = D[(1-\nu)\boldsymbol{\chi} + \nu\operatorname{tr}\boldsymbol{\chi}\,\mathbf{I}]`
and :math:`\mathbf{Q} = K_s G H\boldsymbol{\gamma}`, with
:math:`A_m = EH/(1-\nu^2)` and :math:`D = EH^3/[12(1-\nu^2)]`.  The drilling
strain :math:`\omega` ties the rotation about the normal to the rotation of the
membrane, as in the continuum [HughesBrezzi1989]_, with the moment
:math:`G H\omega`.  Where elements meet at an angle, this gives the six unknowns
of a node a stiffness.  The virtual work of the four resultants gives the flux
and the source of each global equation.  Trusses and shells are linear elastic
and static.

The verification cases of solid mechanics, with their sources, computed values
and errors, are on the :doc:`/verification` page.
