// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Neutronics: the macroscopic cross sections of the multigroup neutron
// diffusion equations,
//
//   -div(D_g grad phi_g) + Sigma_R,g phi_g - sum_{g' != g} Sigma_s,g'->g phi_g'
//       = (chi_g / k) sum_g' nu Sigma_f,g' phi_g',
//
// with the removal cross section Sigma_R,g = Sigma_a,g + sum_{g' != g}
// Sigma_s,g->g' (J. C. Lee, Nuclear Reactor Physics and Engineering, 2nd ed.,
// Wiley 2025, Eqs. 7.18 and 7.19).
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <string>
#include <vector>

namespace dualmesh
{
namespace
{

class MultigroupCrossSections : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "The constant macroscopic cross sections of a material for the multigroup neutron "
        "diffusion equations of the neutron_diffusion physics. For every group g it declares "
        "diffusion_coefficient_g, removal_cross_section_g (the absorption plus the scattering "
        "out of the group), nu_fission_cross_section_g, fission_cross_section_g and "
        "fission_spectrum_g, and for every pair of different groups "
        "scattering_cross_section_h_to_g (from group h into group g) and "
        "fission_production_h_to_g (the fission spectrum of g times the nu-fission cross "
        "section of h). Groups are numbered from 1, the group of highest energy.");
    p.addRequired("diffusion_coefficient",
                  ParameterKind::RealList,
                  "Diffusion coefficient D_g of every group, in m. Each must be positive.");
    p.addRequired("absorption_cross_section",
                  ParameterKind::RealList,
                  "Macroscopic absorption cross section of every group, in 1/m.");
    p.addOptional("scattering_cross_section",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Scattering matrix, in 1/m, given as a list of rows (or as one list, row by "
                  "row): the entry of row h and "
                  "column g is the cross section of scattering from group h into group g. "
                  "The diagonal (scattering within a group) does not enter the equations and "
                  "is ignored. Default none: no scattering between groups.");
    p.addOptional("nu_fission_cross_section",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Number of fission neutrons times the macroscopic fission cross section of "
                  "every group, in 1/m. Default none: a material that does not fission.");
    p.addOptional("fission_cross_section",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Macroscopic fission cross section of every group, in 1/m, for the fission "
                  "rate and the power density. Default none: the nu-fission cross section "
                  "divided by 2.43, the mean number of neutrons per thermal fission of U-235.");
    p.addOptional("fission_spectrum",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Fraction chi_g of the fission neutrons born in every group, which sums to "
                  "one. Default: every fission neutron is born in group 1, the usual "
                  "assumption of few-group thermal reactor calculations.");
    return p;
  }

  explicit MultigroupCrossSections(const InputParameters & p) : Property(p)
  {
    _d = p.getRealList("diffusion_coefficient");
    _a = p.getRealList("absorption_cross_section");
    const std::size_t G = _d.size();
    if (G == 0)
      throw InputError("'" + name() + "': give the diffusion_coefficient of at least one group.");
    auto check = [&](const std::vector<double> & v, const std::string & what, std::size_t n)
    {
      if (!v.empty() && v.size() != n)
        throw InputError("'" + name() + "': " + what + " needs " + std::to_string(n) +
                         " values for " + std::to_string(G) + " groups, and has " +
                         std::to_string(v.size()) + ".");
    };
    check(_a, "absorption_cross_section", G);
    _s = p.getRealList("scattering_cross_section");
    check(_s, "scattering_cross_section", G * G);
    _nf = p.getRealList("nu_fission_cross_section");
    check(_nf, "nu_fission_cross_section", G);
    _f = p.getRealList("fission_cross_section");
    check(_f, "fission_cross_section", G);
    _chi = p.getRealList("fission_spectrum");
    check(_chi, "fission_spectrum", G);
    if (_s.empty())
      _s.assign(G * G, 0.0);
    if (_nf.empty())
      _nf.assign(G, 0.0);
    if (_f.empty())
      for (double v : _nf)
        _f.push_back(v / 2.43);
    if (_chi.empty())
    {
      _chi.assign(G, 0.0);
      _chi[0] = 1.0;
    }
    double chi_sum = 0.0;
    for (std::size_t g = 0; g < G; ++g)
    {
      if (!(_d[g] > 0.0))
        throw InputError("'" + name() + "': the diffusion coefficient of group " +
                         std::to_string(g + 1) + " must be positive.");
      if (_a[g] < 0.0 || _nf[g] < 0.0 || _f[g] < 0.0 || _chi[g] < 0.0)
        throw InputError("'" + name() + "': the cross sections and the fission spectrum of group " +
                         std::to_string(g + 1) + " must not be negative.");
      for (std::size_t h = 0; h < G; ++h)
        if (_s[g * G + h] < 0.0)
          throw InputError("'" + name() + "': the scattering cross sections must not be negative.");
      chi_sum += _chi[g];
    }
    if (std::abs(chi_sum - 1.0) > 1e-6)
      throw InputError("'" + name() + "': the fission spectrum must sum to one, and it sums to " +
                       std::to_string(chi_sum) + ".");
  }

  void declareProperties(PropertyRegistry & r) override
  {
    const std::size_t G = _d.size();
    _ids.clear();
    _values.clear();
    auto add = [&](const std::string & n, double v)
    {
      _ids.push_back(r.declare(n, 1));
      _values.push_back(v);
    };
    for (std::size_t g = 0; g < G; ++g)
    {
      const std::string s = std::to_string(g + 1);
      double removal = _a[g];
      for (std::size_t h = 0; h < G; ++h)
        if (h != g)
          removal += _s[g * G + h];
      add("diffusion_coefficient_" + s, _d[g]);
      add("removal_cross_section_" + s, removal);
      add("nu_fission_cross_section_" + s, _nf[g]);
      add("fission_cross_section_" + s, _f[g]);
      add("fission_spectrum_" + s, _chi[g]);
      for (std::size_t h = 0; h < G; ++h)
        if (h != g)
        {
          const std::string from = std::to_string(h + 1);
          add("scattering_cross_section_" + from + "_to_" + s, _s[h * G + g]);
          add("fission_production_" + from + "_to_" + s, _chi[g] * _nf[h]);
        }
      add("fission_production_" + s + "_to_" + s, _chi[g] * _nf[g]);
    }
  }

  void computeProperties(QpContext & ctx) const override
  {
    for (std::size_t i = 0; i < _ids.size(); ++i)
      ctx.property(_ids[i]) = ADReal(_values[i]);
  }

private:
  std::vector<double> _d, _a, _s, _nf, _f, _chi, _values;
  std::vector<int> _ids;
};

} // namespace

void
registerNeutronicsObjects(Factory & f)
{
  f.add<MultigroupCrossSections>(
      "multigroup_cross_sections", ObjectCategory::Property, "neutronics");
}

} // namespace dualmesh
