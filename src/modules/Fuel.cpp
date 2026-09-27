// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Fuel performance module: the thermal and mechanical properties of nuclear
// fuel (UO2, UN) and cladding (Zircaloy-4), the eigenstrains of fuel
// (thermal expansion, densification, swelling, relocation) and cladding
// (thermal expansion, irradiation growth), and the heat transfer across the
// pellet-cladding gap.  The correlations themselves are in FuelProperties.h,
// with their sources.  The fields that evolve slowly with irradiation
// (burnup, fast flux and fluence, fission rate) enter as functions of space
// and time, which the Python fuel rod driver builds from the power history.
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/FuelProperties.h"

#include <algorithm>
#include <cmath>

namespace dualmesh
{
namespace fuel
{
Gas
gasFromName(const std::string & name)
{
  static const char * names[kNumGases] = {
      "helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"};
  for (int i = 0; i < kNumGases; ++i)
    if (name == names[i])
      return static_cast<Gas>(i);
  throw InputError("Unknown gas '" + name +
                   "' (use helium, argon, krypton, xenon, hydrogen or nitrogen).");
}
} // namespace fuel

namespace
{
using namespace fuel;

// ---------------------------------------------------------------------------
// Burnup, in the unit the user chooses
// ---------------------------------------------------------------------------

/// Joules in one megawatt-day.
constexpr double kJoulesPerMegawattDay = 8.64e10;

/// The parameters of every object that reads a burnup: the burnup itself and
/// the unit it, and every other burnup of the object, is given in.
void
addBurnupParams(InputParameters & p, const std::string & description)
{
  p.addOptional("burnup",
                ParameterKind::Function,
                0.0,
                description + " In the unit named by 'burnup_unit'. Default 0 (fresh fuel).");
  p.addOptional("burnup_unit",
                ParameterKind::String,
                std::string("FIMA"),
                "The unit of every burnup this object takes: 'FIMA' (fissions per initial "
                "heavy-metal atom) or 'MWd/kgHM' (megawatt-days of energy per kilogram of "
                "initial heavy metal). Internally burnup is held in FIMA, and 1 FIMA = E_f N_A / "
                "M_HM, with E_f 'energy_per_fission' and M_HM 'heavy_metal_molar_mass'. "
                "Default FIMA, the unit of the fission rate that the burnup integrates.");
  p.addOptional("energy_per_fission",
                ParameterKind::Real,
                200.0 * 1.602176634e-13,
                "Recoverable energy released per fission, J. Default 3.20435e-11 J (200 MeV), "
                "which with the default molar mass gives 1 FIMA = 938.3 MWd/kgHM, the "
                "conversion of the FRAPCON-4.0 material property report.");
  p.addOptional("heavy_metal_molar_mass",
                ParameterKind::Real,
                0.238029,
                "Molar mass of the heavy metal, kg/mol. Default 0.238029 kg/mol, natural "
                "uranium; enrichment to 5 % changes it by 0.05 %.");
}

/// Converts the burnups of one object to FIMA and from FIMA to MWd/kgHM.
class BurnupUnit
{
public:
  explicit BurnupUnit(const InputParameters & p, const std::string & object)
  {
    const double energy = p.getReal("energy_per_fission");
    const double molar_mass = p.getReal("heavy_metal_molar_mass");
    if (!(energy > 0) || !(molar_mass > 0))
      throw InputError("'" + object +
                       "': energy_per_fission and heavy_metal_molar_mass must be positive.");
    _mwd_per_kg_per_fima = energy * kAvogadro / molar_mass / kJoulesPerMegawattDay;
    const std::string unit = p.getString("burnup_unit");
    if (unit == "FIMA" || unit == "fima")
      _input_is_fima = true;
    else if (unit == "MWd/kgHM" || unit == "mwd/kghm" || unit == "MWd/kgU")
      _input_is_fima = false;
    else
      throw InputError("'" + object + "': unknown burnup_unit '" + unit +
                       "'. Use 'FIMA' or 'MWd/kgHM'.");
  }
  /// A burnup given in the object's unit, in FIMA.
  double toFima(double value) const
  {
    return _input_is_fima ? value : value / _mwd_per_kg_per_fima;
  }
  /// A burnup in FIMA, in MWd/kgHM.
  double toMWdPerKg(double fima) const { return fima * _mwd_per_kg_per_fima; }

private:
  double _mwd_per_kg_per_fima = 938.3;
  bool _input_is_fima = true;
};

/// The theoretical density fraction, with its range checked.
void
addDensityParam(InputParameters & p)
{
  p.addOptional("theoretical_density_fraction",
                ParameterKind::Real,
                0.95,
                "As-fabricated density as a fraction of the theoretical density; the "
                "porosity is one minus it. Default 0.95, typical of LWR fuel.");
}

double
checkedDensity(const InputParameters & p, const std::string & object)
{
  const double density = p.getReal("theoretical_density_fraction");
  if (!(density > 0.5 && density <= 1.0))
    throw InputError("'" + object + "': theoretical_density_fraction must lie in (0.5, 1].");
  return density;
}

void
addTemperatureParam(InputParameters & p)
{
  p.addRequired("temperature", ParameterKind::String, "The temperature variable (kelvin).");
}

// ---------------------------------------------------------------------------
// Thermal properties
// ---------------------------------------------------------------------------

class UO2Thermal : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Thermal properties of UO2 fuel: the material properties 'thermal_conductivity' "
        "(W/m/K), 'specific_heat' (J/kg/K) and 'density' (kg/m^3).");
    addTemperatureParam(p);
    addDensityParam(p);
    addBurnupParams(p, "Local burnup, as a constant or a function of position and time.");
    p.addOptional("thermal_conductivity_model",
                  ParameterKind::String,
                  std::string("fink_lucuta"),
                  "'fink' (Fink 2000, unirradiated, with Fink's porosity correction), "
                  "'fink_lucuta' (Fink's fully dense conductivity times the burnup, porosity "
                  "and radiation factors of Lucuta, Matzke and Hastings 1996), 'nfi' (the "
                  "modified NFI model of FRAPCON-4.0, burnup and gadolinia dependent) or "
                  "'halden' (the Halden correlation that IAEA-TECDOC-1496 recommends for "
                  "irradiated UO2, fitted to Halden in-pile thermocouples). Default "
                  "fink_lucuta: the recommended unirradiated conductivity with a published, "
                  "separable treatment of irradiation.");
    p.addOptional("specific_heat_model",
                  ParameterKind::String,
                  std::string("fink"),
                  "'fink' (Fink 2000) or 'matpro' (Kerrisk and Clifton constants). Default "
                  "fink, the recommendation of IAEA-TECDOC-1496.");
    p.addOptional("gadolinia_weight_fraction",
                  ParameterKind::Real,
                  0.0,
                  "Gd2O3 weight fraction, for the nfi model only. Default 0 (no gadolinia).");
    return p;
  }
  explicit UO2Thermal(const InputParameters & p)
      : Material(p), _burnup_unit(p, name()), _density(checkedDensity(p, name()))
  {
    const auto m = p.getString("thermal_conductivity_model");
    if (m == "fink")
      _model = 0;
    else if (m == "fink_lucuta")
      _model = 1;
    else if (m == "nfi")
      _model = 2;
    else if (m == "halden")
      _model = 3;
    else
      throw InputError("'" + name() + "': unknown thermal_conductivity_model '" + m +
                       "'. Use fink, fink_lucuta, nfi or halden.");
    const auto c = p.getString("specific_heat_model");
    if (c != "fink" && c != "matpro")
      throw InputError("'" + name() + "': unknown specific_heat_model '" + c +
                       "'. Use fink or matpro.");
    _matpro_cp = c == "matpro";
    _gadolinia = p.getReal("gadolinia_weight_fraction");
    if (_gadolinia != 0.0 && _model != 2)
      throw InputError("'" + name() +
                       "': gadolinia_weight_fraction is used by the nfi model only; set "
                       "thermal_conductivity_model = nfi or leave it at 0.");
  }
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
    _burnup = getFunction(problem, "burnup");
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _k = r.declare("thermal_conductivity", 1);
    _cp = r.declare("specific_heat", 1);
    _rho = r.declare("density", 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    const double fima = std::max(0.0, _burnup_unit.toFima(_burnup->value(ctx.x, ctx.time)));
    const double porosity = 1.0 - _density;
    ADReal k;
    if (_model == 0)
      k = uo2ConductivityFink100(T) * (1.0 - (2.6 - 0.5 * T / 1000.0) * porosity);
    else if (_model == 1)
      k = uo2ConductivityFinkLucuta(T, 100.0 * fima, porosity);
    else if (_model == 2)
      k = uo2ConductivityNFI(T, _burnup_unit.toMWdPerKg(fima), _gadolinia, _density);
    else
      // MWd per kg of heavy metal to MWd per kg of UO2.
      k = uo2ConductivityHalden(T,
                                _burnup_unit.toMWdPerKg(fima) * (kMolarMassUO2 - 2 * 15.999e-3) /
                                    kMolarMassUO2,
                                _density);
    ctx.property(_k) = k;
    ctx.property(_cp) = _matpro_cp ? uo2SpecificHeatMATPRO(T) : uo2SpecificHeatFink(T);
    ctx.property(_rho) = ADReal(kDensityUO2 * _density);
  }

private:
  BurnupUnit _burnup_unit;
  double _density;
  int _model = 1;
  bool _matpro_cp = false;
  double _gadolinia = 0.0;
  int _T = -1, _k = -1, _cp = -1, _rho = -1;
  FunctionPtr _burnup;
};

class UNThermal : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Thermal properties of uranium mononitride fuel from Hayes, Thomas and Peddicord "
        "(1990): 'thermal_conductivity' = 1.864 exp(-2.14 P) T^0.361 W/m/K with the porosity "
        "P, 'specific_heat' (Einstein temperature 365.7 K) and 'density'. The conductivity "
        "does not depend on burnup; no public, peer-reviewed burnup dependence was found, "
        "so this object takes no burnup.");
    addTemperatureParam(p);
    addDensityParam(p);
    return p;
  }
  explicit UNThermal(const InputParameters & p) : Material(p), _density(checkedDensity(p, name()))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _k = r.declare("thermal_conductivity", 1);
    _cp = r.declare("specific_heat", 1);
    _rho = r.declare("density", 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    ctx.property(_k) = unConductivityHayes(T, 1.0 - _density);
    ctx.property(_cp) = unSpecificHeatHayes(T);
    ctx.property(_rho) = ADReal(kDensityUN * _density);
  }

private:
  double _density;
  int _T = -1, _k = -1, _cp = -1, _rho = -1;
};

class ZircaloyThermal : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Thermal properties of Zircaloy-2 and Zircaloy-4 cladding: 'thermal_conductivity' "
        "(IAEA-TECDOC-1496, 12.767 - 5.4348e-4 T + 8.9818e-6 T^2, 300-1800 K), "
        "'specific_heat' (IAEA-TECDOC-1496, with the alpha-beta transition peak) and "
        "'density' (alpha phase at 300 K, held constant).");
    addTemperatureParam(p);
    return p;
  }
  explicit ZircaloyThermal(const InputParameters & p) : Material(p) {}
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _k = r.declare("thermal_conductivity", 1);
    _cp = r.declare("specific_heat", 1);
    _rho = r.declare("density", 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    ctx.property(_k) = zryConductivity(T);
    ctx.property(_cp) = zrySpecificHeat(T);
    ctx.property(_rho) = ADReal(zryDensity(300.0));
  }

private:
  int _T = -1, _k = -1, _cp = -1, _rho = -1;
};

// ---------------------------------------------------------------------------
// Elastic constants
// ---------------------------------------------------------------------------

/// Base of the elasticity materials: declares 'youngs_modulus' and
/// 'poissons_ratio'.
class ElasticityBase : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    addTemperatureParam(p);
    return p;
  }
  using Material::Material;
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _E = r.declare("youngs_modulus", 1);
    _nu = r.declare("poissons_ratio", 1);
  }

protected:
  int _T = -1, _E = -1, _nu = -1;
};

class UO2Elasticity : public ElasticityBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = ElasticityBase::validParams();
    p.setClassDescription(
        "Elastic constants of UO2 as the material properties 'youngs_modulus' (Pa) and "
        "'poissons_ratio': MATPRO FELMOD, E = 2.334e11 (1 - 1.0915e-4 T)(1 - 2.752 P), and "
        "FPOIR, nu = 0.316.");
    addDensityParam(p);
    return p;
  }
  explicit UO2Elasticity(const InputParameters & p)
      : ElasticityBase(p), _density(checkedDensity(p, name()))
  {
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    ctx.property(_E) = uo2YoungsModulus(T, _density);
    ctx.property(_nu) = ADReal(kPoissonRatioUO2);
  }

private:
  double _density;
};

class UNElasticity : public ElasticityBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = ElasticityBase::validParams();
    p.setClassDescription("Elastic constants of UN as 'youngs_modulus' (Pa) and "
                          "'poissons_ratio', from Hayes, Thomas and Peddicord (1990).");
    addDensityParam(p);
    return p;
  }
  explicit UNElasticity(const InputParameters & p)
      : ElasticityBase(p), _density(checkedDensity(p, name()))
  {
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    ctx.property(_E) = unYoungsModulus(T, 100.0 * _density);
    ctx.property(_nu) = ADReal(unPoissonRatio(100.0 * _density));
  }

private:
  double _density;
};

class ZircaloyElasticity : public ElasticityBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = ElasticityBase::validParams();
    p.setClassDescription(
        "Elastic constants of Zircaloy as 'youngs_modulus' (Pa) and 'poissons_ratio' "
        "(MATPRO CELMOD and CSHEAR, with the cold-work and fast neutron fluence factors; nu "
        "from E and the shear modulus).");
    p.addOptional("cold_work",
                  ParameterKind::Real,
                  0.0,
                  "Cold work of the cladding, as a ratio of areas (MATPRO's K2 = -2.6e10 C Pa on "
                  "both moduli). Default 0 (recrystallised); stress-relieved tubing keeps part of "
                  "its cold work.");
    p.addOptional("fast_neutron_fluence",
                  ParameterKind::Function,
                  0.0,
                  "Fast neutron fluence (E > 1 MeV), n/m^2. Default 0 (unirradiated).");
    return p;
  }
  explicit ZircaloyElasticity(const InputParameters & p) : ElasticityBase(p)
  {
    _cold_work = p.getReal("cold_work");
    if (!(_cold_work >= 0.0 && _cold_work < 1.0))
      throw InputError("'" + name() + "': cold_work must lie in [0, 1).");
  }
  void initialSetup(Problem & problem) override
  {
    ElasticityBase::initialSetup(problem);
    _fluence = getFunction(problem, "fast_neutron_fluence");
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal T = correlationTemperature(ctx.coefficientValue(_T));
    const double phi = _fluence->value(ctx.x, ctx.time);
    const ADReal E = zryYoungsModulus(T, phi, _cold_work);
    ctx.property(_E) = E;
    ctx.property(_nu) = E / (2.0 * zryShearModulus(T, phi, _cold_work)) - 1.0;
  }

private:
  FunctionPtr _fluence;
  double _cold_work = 0.0;
};

// ---------------------------------------------------------------------------
// Eigenstrains
// ---------------------------------------------------------------------------

/// The index of the axial normal strain component in the Voigt storage (xx,
/// yy, zz, yz, xz, xy) for each formulation: axisymmetric (r, z, theta) with
/// the axis in slot 1; a three-dimensional rod along z with the axis in slot 2.
int
axialSlot(const std::string & formulation)
{
  if (formulation == "axisymmetric" || formulation == "axisymmetric_1d")
    return 1;
  if (formulation == "three_dimensional")
    return 2;
  throw InputError("Unknown formulation '" + formulation +
                   "'. Use axisymmetric, axisymmetric_1d or three_dimensional.");
}

/// Base of the eigenstrain materials: declares one six-component property.
class EigenstrainBase : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.addRequired("eigenstrain_name",
                  ParameterKind::String,
                  "Name of the material property (six Voigt components) this eigenstrain is "
                  "stored in; list it in the 'eigenstrain_names' of the stress material.");
    p.addRequired("formulation",
                  ParameterKind::String,
                  "The formulation of the stress material it feeds: axisymmetric (an r, z "
                  "mesh), axisymmetric_1d (a radial slice in generalized plane strain) or "
                  "three_dimensional (a rod along z).");
    return p;
  }
  explicit EigenstrainBase(const InputParameters & p) : Material(p)
  {
    _axial = axialSlot(p.getString("formulation"));
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("eigenstrain_name"), 6);
  }

protected:
  /// Store an eigenstrain with the given axial and transverse normal parts.
  void store(QpContext & ctx, const ADReal & axial, const ADReal & transverse) const
  {
    for (int i = 0; i < 6; ++i)
      ctx.property(_prop, i) = ADReal(0.0);
    for (int i = 0; i < 3; ++i)
      ctx.property(_prop, i) = i == _axial ? axial : transverse;
  }
  /// Store a strain that acts in the plane normal to the axis only (radial and
  /// hoop in an axisymmetric body, x and y in three dimensions).
  void storeTransverse(QpContext & ctx, const ADReal & e) const { store(ctx, ADReal(0.0), e); }

  int _axial = 1;
  int _prop = -1;
};

/// Thermal expansion of one material, from its stress-free temperature.
template <int Material_> class ThermalExpansionEigenstrainOf : public EigenstrainBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = EigenstrainBase::validParams();
    if (Material_ == 0)
      p.setClassDescription("Thermal expansion strain of UO2 (Martin 1988), isotropic, from "
                            "the stress-free temperature.");
    else if (Material_ == 1)
      p.setClassDescription("Thermal expansion strain of UN (Hayes et al. 1990), isotropic, "
                            "from the stress-free temperature.");
    else
      p.setClassDescription(
          "Thermal expansion strain of alpha-phase Zircaloy cladding of unknown texture "
          "(IAEA-TECDOC-1496 Eqs. 4-6, after Bunnell et al.): hoop 7.092e-6, axial 5.458e-6 "
          "and radial 9.999e-6 per K, from the stress-free temperature. In three dimensions "
          "the rod axis is the z axis through the origin, and the radial and hoop strains are "
          "rotated to x and y at each point.");
    addTemperatureParam(p);
    p.addOptional("stress_free_temperature",
                  ParameterKind::Real,
                  293.15,
                  "Temperature at which the thermal strain is zero, K. Default 293.15 K, room "
                  "temperature, at which fuel and cladding are fabricated and measured.");
    return p;
  }
  explicit ThermalExpansionEigenstrainOf(const InputParameters & p) : EigenstrainBase(p)
  {
    _Tref = p.getReal("stress_free_temperature");
  }
  void initialSetup(Problem & problem) override
  {
    EigenstrainBase::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal & T = ctx.value(_T);
    if constexpr (Material_ == 0)
    {
      const ADReal e = uo2ThermalStrainMartin(T, _Tref);
      store(ctx, e, e);
    }
    else if constexpr (Material_ == 1)
    {
      const ADReal e = unThermalStrainHayes(T, _Tref);
      store(ctx, e, e);
    }
    else
    {
      const ADReal axial = zryAxialThermalStrain(T, _Tref);
      const ADReal hoop = zryHoopThermalStrain(T, _Tref);
      const ADReal radial = zryRadialThermalStrain(T, _Tref);
      for (int i = 0; i < 6; ++i)
        ctx.property(_prop, i) = ADReal(0.0);
      if (_axial == 1)
      {
        // Axisymmetric (r, z, theta).
        ctx.property(_prop, 0) = radial;
        ctx.property(_prop, 1) = axial;
        ctx.property(_prop, 2) = hoop;
      }
      else
      {
        // A rod along z: rotate the radial and hoop strains to x and y, with
        // the engineering shear strain gamma_xy in slot 5.
        const double r = std::hypot(ctx.x[0], ctx.x[1]);
        const double c = r > 0 ? ctx.x[0] / r : 1.0, sn = r > 0 ? ctx.x[1] / r : 0.0;
        ctx.property(_prop, 0) = radial * (c * c) + hoop * (sn * sn);
        ctx.property(_prop, 1) = radial * (sn * sn) + hoop * (c * c);
        ctx.property(_prop, 2) = axial;
        ctx.property(_prop, 5) = 2.0 * (radial - hoop) * (c * sn);
      }
    }
  }

private:
  double _Tref = 293.15;
  int _T = -1;
};
using UO2ThermalExpansionEigenstrain = ThermalExpansionEigenstrainOf<0>;
using UNThermalExpansionEigenstrain = ThermalExpansionEigenstrainOf<1>;
using ZircaloyThermalExpansionEigenstrain = ThermalExpansionEigenstrainOf<2>;

class UO2VolumetricSwellingEigenstrain : public EigenstrainBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = EigenstrainBase::validParams();
    p.setClassDescription(
        "Volumetric strain of UO2 fuel from irradiation: densification (MATPRO FUDENS or "
        "the ESCORE form), solid fission-product swelling (MATPRO, 5.577e-5 rho per FIMA) and "
        "an optional element field holding the accumulated gaseous swelling. One third of "
        "the volumetric strain is applied to each normal component. Densification is "
        "evaluated at the highest temperature the fuel has reached (an element field kept "
        "by the rod driver, or the current temperature if none is given), so that it never "
        "reverses when the fuel cools.");
    addTemperatureParam(p);
    addDensityParam(p);
    addBurnupParams(p, "Local burnup.");
    p.addOptional("total_densification",
                  ParameterKind::Real,
                  0.01,
                  "Total densification, as a fraction of the theoretical density: the density "
                  "change in a resintering test (1973 K for 24 h). The matpro model takes the "
                  "resintering density change RSNTR = total_densification times the theoretical "
                  "density; the escore model takes it as the final volumetric shrinkage. "
                  "Default 0.01, a typical value for LWR fuel; set it from the fuel's own test.");
    p.addOptional("densification_model",
                  ParameterKind::String,
                  std::string("matpro"),
                  "'matpro' (FUDENS, NUREG/CR-6150 Vol. 4 Eqs. 2-81, 2-82 and 2-85, read in the "
                  "report) or 'escore' (the ESCORE form of Rashid et al., EPRI 1011308, which is "
                  "not public; taken from secondary documentation). Default matpro, the model "
                  "whose source could be read.");
    p.addOptional("maximum_temperature_field",
                  ParameterKind::String,
                  std::string(""),
                  "Name of an element field holding the highest temperature each element has "
                  "reached, K, kept by the rod driver. The densification uses the larger of it "
                  "and the current temperature. Default empty (the current temperature only).");
    p.addOptional("densification_complete_burnup",
                  ParameterKind::Real,
                  5.0,
                  "Burnup at which densification is complete, in MWd/kgHM whatever "
                  "'burnup_unit' is; escore model only. Default 5 MWd/kgHM, the ESCORE value.");
    p.addOptional("include_solid_swelling",
                  ParameterKind::Boolean,
                  true,
                  "Include the solid fission-product swelling. Default true.");
    p.addOptional("gaseous_swelling_field",
                  ParameterKind::String,
                  std::string(""),
                  "Name of an element field holding the accumulated gaseous swelling "
                  "(volumetric strain), integrated between time steps by the rod driver. "
                  "Default empty (no gaseous swelling).");
    return p;
  }
  explicit UO2VolumetricSwellingEigenstrain(const InputParameters & p)
      : EigenstrainBase(p), _burnup_unit(p, name()), _density(checkedDensity(p, name()))
  {
    _total = p.getReal("total_densification");
    _complete = p.getReal("densification_complete_burnup");
    _solid = p.getBool("include_solid_swelling");
    const auto model = p.getString("densification_model");
    if (model != "matpro" && model != "escore")
      throw InputError("'" + name() + "': unknown densification_model '" + model +
                       "'. Use matpro or escore.");
    _escore = model == "escore";
  }
  void initialSetup(Problem & problem) override
  {
    EigenstrainBase::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
    _burnup = getFunction(problem, "burnup");
    const auto field = _params.getString("gaseous_swelling_field");
    _gas = field.empty() ? nullptr : &problem.elementField(field);
    const auto tmax = _params.getString("maximum_temperature_field");
    _Tmax = tmax.empty() ? nullptr : &problem.elementField(tmax);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const double fima = std::max(0.0, _burnup_unit.toFima(_burnup->value(ctx.x, ctx.time)));
    ADReal T = correlationTemperature(ctx.value(_T));
    if (_Tmax && ctx.element >= 0 && (*_Tmax)[ctx.element] > T.value())
      T = ADReal((*_Tmax)[ctx.element]);
    const double mwd = _burnup_unit.toMWdPerKg(fima);
    ADReal volumetric =
        _escore ? uo2DensificationESCORE(T, mwd, _total, _complete)
                : ADReal(3.0 * uo2DensificationMATPRO(T.value(), mwd, _total * kDensityUO2));
    if (_solid)
      volumetric += uo2SolidSwellingRate(kDensityUO2 * _density) * fima;
    if (_gas && ctx.element >= 0)
      volumetric += (*_gas)[ctx.element];
    const ADReal e = volumetric / 3.0;
    store(ctx, e, e);
  }

private:
  BurnupUnit _burnup_unit;
  double _density;
  double _total = 0.01, _complete = 5.0;
  bool _solid = true, _escore = false;
  int _T = -1;
  FunctionPtr _burnup;
  const std::vector<double> * _gas = nullptr;
  const std::vector<double> * _Tmax = nullptr;
};

class UNVolumetricSwellingEigenstrain : public EigenstrainBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = EigenstrainBase::validParams();
    p.setClassDescription(
        "Volumetric swelling of UN fuel from the correlation of S. B. Ross, El-Genk and "
        "Matthews (1990), 4.7e-11 T^3.12 B^0.83 rho^0.5 per cent, bounded below by 1 per cent "
        "per atom per cent of burnup, the lowest swelling rate measured for nitride fuel "
        "(NEA No. 7317, 2018). It is evaluated with the local temperature and burnup, one "
        "third on each normal component. The correlation gives the total swelling at an "
        "average temperature; applying it locally is a modelling choice.");
    addTemperatureParam(p);
    addDensityParam(p);
    addBurnupParams(p, "Local burnup.");
    return p;
  }
  explicit UNVolumetricSwellingEigenstrain(const InputParameters & p)
      : EigenstrainBase(p), _burnup_unit(p, name()), _density(checkedDensity(p, name()))
  {
  }
  void initialSetup(Problem & problem) override
  {
    EigenstrainBase::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
    _burnup = getFunction(problem, "burnup");
  }
  void computeProperties(QpContext & ctx) const override
  {
    const double at_percent =
        100.0 * std::max(0.0, _burnup_unit.toFima(_burnup->value(ctx.x, ctx.time)));
    const ADReal T = correlationTemperature(ctx.value(_T));
    ADReal e(0.0);
    if (at_percent > 0)
    {
      e = 0.01 / 3.0 * 4.7e-11 * pow(T, 3.12) * std::pow(at_percent, 0.83) *
          std::sqrt(100.0 * _density);
      const double floor = 0.01 / 3.0 * kUNMinimumSwellingPercentPerAtomPercent * at_percent;
      if (e.value() < floor)
        e = ADReal(floor);
    }
    store(ctx, e, e);
  }

private:
  BurnupUnit _burnup_unit;
  double _density;
  int _T = -1;
  FunctionPtr _burnup;
};

class UO2RelocationEigenstrain : public EigenstrainBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = EigenstrainBase::validParams();
    p.setClassDescription(
        "Relocation of cracked fuel fragments towards the cladding, as a strain in the plane "
        "normal to the rod axis, in the ESCORE form documented in the BISON theory manual "
        "(Hales et al. 2013): dD/D = 0.80 Q (G0/D0)(0.005 Bu^0.3 - 0.20 D0 + 0.3) with the "
        "linear heat rate q' in kW/ft (Q = 0 below 6, (q' - 6)^(1/3) up to 14 and (q' - 10)/2 "
        "above), the cold diametral gap G0 and pellet diameter D0 in inches and the burnup "
        "in MWd/tU. The same strain is applied radially and circumferentially, which moves "
        "the fragments outwards without stressing them.");
    p.addRequired("linear_heat_rate",
                  ParameterKind::Function,
                  "Rod-average linear heat rate at the axial position, W/m.");
    addBurnupParams(p, "Pellet-average burnup.");
    p.addRequired("pellet_diameter", ParameterKind::Real, "Cold pellet diameter, m.");
    p.addRequired("diametral_gap", ParameterKind::Real, "Cold diametral gap, m.");
    p.addOptional("relocation_burnup_limit",
                  ParameterKind::Real,
                  11.5,
                  "Burnup, in MWd/kgHM whatever 'burnup_unit' is, above which relocation no "
                  "longer grows. Default 11.5 MWd/kgHM, where the correlation's stated range "
                  "ends.");
    return p;
  }
  explicit UO2RelocationEigenstrain(const InputParameters & p)
      : EigenstrainBase(p), _burnup_unit(p, name())
  {
    _D0 = p.getReal("pellet_diameter");
    _G0 = p.getReal("diametral_gap");
    _limit = p.getReal("relocation_burnup_limit");
  }
  void initialSetup(Problem & problem) override
  {
    EigenstrainBase::initialSetup(problem);
    _q = getFunction(problem, "linear_heat_rate");
    _burnup = getFunction(problem, "burnup");
  }
  void computeProperties(QpContext & ctx) const override
  {
    const double mwd_per_kg = _burnup_unit.toMWdPerKg(
        std::max(0.0, _burnup_unit.toFima(_burnup->value(ctx.x, ctx.time))));
    storeTransverse(
        ctx, ADReal(uo2RelocationStrain(_q->value(ctx.x, ctx.time), mwd_per_kg, _D0, _G0, _limit)));
  }

private:
  BurnupUnit _burnup_unit;
  double _D0 = 0, _G0 = 0, _limit = 11.5;
  FunctionPtr _q, _burnup;
};

class ZircaloyIrradiationGrowthEigenstrain : public EigenstrainBase
{
public:
  static InputParameters validParams()
  {
    InputParameters p = EigenstrainBase::validParams();
    p.setClassDescription(
        "Axial irradiation growth of stress-relieved Zircaloy-4 (Franklin 1982), with the "
        "volume conserved by an equal contraction of the two transverse directions.");
    p.addRequired("fast_neutron_fluence",
                  ParameterKind::Function,
                  "Fast neutron fluence (E > 1 MeV), n/m^2.");
    return p;
  }
  explicit ZircaloyIrradiationGrowthEigenstrain(const InputParameters & p) : EigenstrainBase(p) {}
  void initialSetup(Problem & problem) override
  {
    EigenstrainBase::initialSetup(problem);
    _fluence = getFunction(problem, "fast_neutron_fluence");
  }
  void computeProperties(QpContext & ctx) const override
  {
    const double axial = zryIrradiationGrowth(_fluence->value(ctx.x, ctx.time));
    store(ctx, ADReal(axial), ADReal(1.0 / std::sqrt(1.0 + axial) - 1.0));
  }

private:
  FunctionPtr _fluence;
};

// ---------------------------------------------------------------------------
// The pellet-cladding gap
// ---------------------------------------------------------------------------

class GasGapHeatTransfer : public InterfaceBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = InterfaceBC::validParams();
    p.setClassDescription(
        "Heat transfer across a gas-filled gap, such as the pellet-cladding gap of a fuel "
        "rod: gas conduction, "
        "radiation and solid contact, h_gap = h_gas + h_rad + h_solid (Ross and Stoute "
        "1962). The gas conductance is k_gas / (g + C_r (R_p + R_s) + j), with the width g "
        "of the gap measured along the primary normal between the displaced surfaces, the "
        "roughnesses R, the roughness coefficient C_r and the temperature jump distance j "
        "(Kennard; Lanning and Hann 1975). k_gas is the Lindsay-Bromley/Brokaw mixture "
        "conductivity of the fill and fission gases (MATPRO fits). Radiation is between two "
        "parallel grey surfaces. The solid contact conductance, used when a contact penalty "
        "is given and the gap is closed, is C_s k_m P_c / (sqrt(delta) H) with the harmonic "
        "mean conductivity k_m, the contact pressure P_c, the Meyer hardness H and delta = "
        "0.8 (R_p + R_s) (Ross-Stoute form of the BISON theory manual, 2013). Temperatures "
        "must be in kelvin.");
    p.addOptional("displacements",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Displacement variables. Default none: the gap keeps its as-meshed width.");
    deformation::addParameter(p,
                              "the conductance is per unit deformed primary area, and the flux "
                              "is multiplied by the area ratio da/dA = |J F^-T N| of the "
                              "primary surface.");
    for (const char * gas : {"helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"})
      p.addOptional(
          std::string(gas) + "_fraction",
          ParameterKind::Function,
          std::string(gas) == "helium" ? 1.0 : 0.0,
          std::string("Mole fraction of ") + gas + " in the gap gas. Default " +
              (std::string(gas) == "helium" ? "1 (pure helium, the usual fill gas)." : "0."));
    p.addRequired("gas_pressure", ParameterKind::Function, "Pressure of the gap gas, Pa.");
    p.addOptional("primary_roughness",
                  ParameterKind::Real,
                  2.0e-6,
                  "Roughness of the primary surface, m. Default 2 um, typical of a ground "
                  "UO2 pellet.");
    p.addOptional("secondary_roughness",
                  ParameterKind::Real,
                  1.0e-6,
                  "Roughness of the secondary surface, m. Default 1 um, typical of the inner "
                  "surface of a cladding tube.");
    p.addOptional("roughness_coefficient",
                  ParameterKind::Real,
                  1.5,
                  "C_r, the factor on the summed roughnesses in the gas conductance. Default "
                  "1.5, the Ross-Stoute value.");
    p.addOptional("primary_emissivity",
                  ParameterKind::Real,
                  0.8,
                  "Emissivity of the primary surface. Default 0.8, the MATPRO value for UO2.");
    p.addOptional("secondary_emissivity",
                  ParameterKind::Real,
                  0.8,
                  "Emissivity of the secondary surface. Default 0.8, typical of oxidised "
                  "Zircaloy.");
    p.addOptional("contact_penalty",
                  ParameterKind::Real,
                  0.0,
                  "Penalty stiffness of the mechanical contact (Pa/m), the same value as the "
                  "'penalty' of the gap_contact objects. Default 0, which disables the solid "
                  "contact conductance (for a problem without mechanics).");
    p.addOptional("meyer_hardness",
                  ParameterKind::Real,
                  6.8e8,
                  "Meyer hardness H of the softer surface, Pa, with meyer_hardness_model = "
                  "constant. Default 0.68 GPa, Zircaloy near 600 K.");
    p.addOptional("meyer_hardness_model",
                  ParameterKind::String,
                  std::string("constant"),
                  "'constant' (meyer_hardness) or 'zircaloy' (MATPRO CMHARD at the temperature of "
                  "the secondary surface, from 2 GPa at room temperature to 0.2 GPa at 875 K). "
                  "Default constant.");
    p.addOptional("solid_contact_coefficient",
                  ParameterKind::Real,
                  10.0,
                  "C_s of the solid contact conductance, m^(-1/2). Default 10, the "
                  "Ross-Stoute value.");
    p.addOptional("primary_conductivity",
                  ParameterKind::Function,
                  3.0,
                  "Conductivity of the primary body at its surface (W/m/K), used by the solid "
                  "contact term only when no 'thermal_conductivity' material property exists. "
                  "Default 3 W/m/K, UO2 near 1000 K.");
    p.addOptional("secondary_conductivity",
                  ParameterKind::Function,
                  17.0,
                  "Conductivity of the secondary body at its surface (W/m/K), for the solid "
                  "contact term. Default 17 W/m/K, Zircaloy near 600 K.");
    return p;
  }
  explicit GasGapHeatTransfer(const InputParameters & p) : InterfaceBC(p)
  {
    _Rp = p.getReal("primary_roughness");
    _Rs = p.getReal("secondary_roughness");
    _Cr = p.getReal("roughness_coefficient");
    const double ep = p.getReal("primary_emissivity"), es = p.getReal("secondary_emissivity");
    _F = ep > 0 && es > 0 ? 1.0 / (1.0 / ep + 1.0 / es - 1.0) : 0.0;
    _penalty = p.getReal("contact_penalty");
    _H = p.getReal("meyer_hardness");
    const auto hm = p.getString("meyer_hardness_model");
    if (hm != "constant" && hm != "zircaloy")
      throw InputError("'" + name() + "': meyer_hardness_model must be constant or zircaloy.");
    _zircaloy_hardness = hm == "zircaloy";
    _Cs = p.getReal("solid_contact_coefficient");
  }
  void initialSetup(Problem & problem) override
  {
    InterfaceBC::initialSetup(problem);
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
    {
      _disp.push_back(problem.variableIndex(d));
      addCoupled(_disp.back());
    }
    int i = 0;
    for (const char * gas : {"helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"})
      _x[i++] = getFunction(problem, std::string(gas) + "_fraction");
    _P = getFunction(problem, "gas_pressure");
    _kp = getFunction(problem, "primary_conductivity");
    _ks = getFunction(problem, "secondary_conductivity");
    _kprop = problem.propertyRegistry().has("thermal_conductivity")
                 ? problem.propertyRegistry().id("thermal_conductivity")
                 : -1;
    _Fprop = deformation::propertyId(problem, _params, name());
  }
  /// Gap width along the primary normal between the displaced surfaces.
  ADReal gapWidth(const QpContext & ctx) const
  {
    ADReal g(0.0);
    for (int d = 0; d < ctx.dim; ++d)
    {
      ADReal diff(ctx.x_other[d] - ctx.x[d]);
      if (d < static_cast<int>(_disp.size()))
        diff = diff + ctx.u_other[_disp[d]] - ctx.u[_disp[d]];
      g += diff * ctx.normal[d];
    }
    return g;
  }
  ADReal computeInterfaceFlux(const QpContext & ctx) const override
  {
    const ADReal & Tp = ctx.value(_var);
    const ADReal & Ts = ctx.u_other[_secondary_var];
    double x[kNumGases];
    double sum = 0;
    for (int i = 0; i < kNumGases; ++i)
    {
      x[i] = std::max(0.0, _x[i]->value(ctx.x, ctx.time));
      sum += x[i];
    }
    if (sum <= 0)
      throw InputError("'" + name() + "': the gap gas mole fractions add up to zero.");
    for (double & xi : x)
      xi /= sum;
    // The gas temperature, floored for the gas correlations (see
    // correlationTemperature); the radiation term uses the true temperatures.
    const ADReal Tg = correlationTemperature(ADReal(0.5 * (Tp + Ts)));
    const double P = _P->value(ctx.x, ctx.time);
    const ADReal kg = gasMixtureConductivity(x, Tg);
    const ADReal jump = gasJumpDistance(x, Tg, P, kg);
    const ADReal g = gapWidth(ctx);
    const ADReal open = g.value() > 0 ? g : ADReal(0.0);
    ADReal h = kg / (open + _Cr * (_Rp + _Rs) + jump);
    if (_F > 0)
      h += kStefanBoltzmann * _F * (Tp * Tp + Ts * Ts) * (Tp + Ts);
    if (_penalty > 0 && g.value() < 0)
    {
      const ADReal kp = _kprop >= 0 ? ctx.property(_kprop) : ADReal(_kp->value(ctx.x, ctx.time));
      const double ks = _ks->value(ctx.x, ctx.time);
      const ADReal km = 2.0 * kp * ks / (kp + ks);
      const ADReal Pc = -_penalty * g;
      const ADReal H = _zircaloy_hardness ? zryMeyerHardness(Ts) : ADReal(_H);
      h += _Cs * km * Pc / (std::sqrt(0.8 * (_Rp + _Rs)) * H);
    }
    const ADReal q = h * (Ts - Tp);
    return _Fprop >= 0 ? q * deformation::areaRatio(ctx, _Fprop) : q;
  }

private:
  int _Fprop = -1;
  std::vector<int> _disp;
  FunctionPtr _x[kNumGases];
  FunctionPtr _P, _kp, _ks;
  int _kprop = -1;
  double _Rp = 2e-6, _Rs = 1e-6, _Cr = 1.5, _F = 0.0, _penalty = 0.0, _H = 6.8e8, _Cs = 10.0;
  bool _zircaloy_hardness = false;
};

} // namespace

void
registerFuelObjects(Factory & f)
{
  const std::string m = "fuel_performance";
  f.add<UO2Thermal>("UO2_thermal", ObjectCategory::Material, m);
  f.add<UNThermal>("UN_thermal", ObjectCategory::Material, m);
  f.add<ZircaloyThermal>("Zircaloy_thermal", ObjectCategory::Material, m);
  f.add<UO2Elasticity>("UO2_elasticity", ObjectCategory::Material, m);
  f.add<UNElasticity>("UN_elasticity", ObjectCategory::Material, m);
  f.add<ZircaloyElasticity>("Zircaloy_elasticity", ObjectCategory::Material, m);
  f.add<UO2ThermalExpansionEigenstrain>(
      "UO2_thermal_expansion_eigenstrain", ObjectCategory::Material, m);
  f.add<UNThermalExpansionEigenstrain>(
      "UN_thermal_expansion_eigenstrain", ObjectCategory::Material, m);
  f.add<ZircaloyThermalExpansionEigenstrain>(
      "Zircaloy_thermal_expansion_eigenstrain", ObjectCategory::Material, m);
  f.add<UO2VolumetricSwellingEigenstrain>(
      "UO2_volumetric_swelling_eigenstrain", ObjectCategory::Material, m);
  f.add<UNVolumetricSwellingEigenstrain>(
      "UN_volumetric_swelling_eigenstrain", ObjectCategory::Material, m);
  f.add<UO2RelocationEigenstrain>("UO2_relocation_eigenstrain", ObjectCategory::Material, m);
  f.add<ZircaloyIrradiationGrowthEigenstrain>(
      "Zircaloy_irradiation_growth_eigenstrain", ObjectCategory::Material, m);
  f.add<GasGapHeatTransfer>("gas_gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
