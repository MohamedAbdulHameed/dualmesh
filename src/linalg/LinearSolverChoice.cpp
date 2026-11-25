// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/LinearSolverChoice.h"

#include "dualmesh/core/InputParameters.h"

#include <cstdlib>
#include <sstream>

namespace dualmesh
{

namespace
{
/// Whether @p word names an option: it starts with a dash and is not a number.
bool
isOptionName(const std::string & word)
{
  if (word.size() < 2 || word[0] != '-')
    return false;
  char * end = nullptr;
  std::strtod(word.c_str(), &end);
  return *end != '\0';
}
} // namespace

user_options_parts
user_choice(const std::string & options)
{
  std::istringstream words(options);
  std::vector<std::string> list;
  for (std::string word; words >> word;)
    list.push_back(word);
  user_options_parts parts;
  const auto append = [](std::string & text, const std::string & word)
  { text += (text.empty() ? "" : " ") + word; };
  std::string * target = &parts.linear;
  for (std::size_t k = 0; k < list.size(); ++k)
  {
    const std::string & word = list[k];
    if (!isOptionName(word))
    {
      append(*target, word);
      continue;
    }
    const bool has_value = k + 1 < list.size() && !isOptionName(list[k + 1]);
    const std::string value = has_value ? list[k + 1] : "";
    if (word == "-dualmesh_schur")
    {
      if (!has_value || !(parts.split.Schur = schurApproximation(value)))
        throw InputError("The PETSc option -dualmesh_schur takes lsc, pcd or pressure_mass, the "
                         "approximation of the Schur complement of a flow" +
                         (value.empty() ? std::string(".") : ", not '" + value + "'."));
      ++k;
      continue;
    }
    if (word == "-dualmesh_split")
    {
      if (!has_value || value != "physics")
        throw InputError("The PETSc option -dualmesh_split takes physics, the split of the system "
                         "by physics" +
                         (value.empty() ? std::string(".") : ", not '" + value + "'."));
      parts.split.physics = true;
      ++k;
      continue;
    }
    target = word.rfind("-snes_", 0) == 0 ? &parts.nonlinear : &parts.linear;
    append(*target, word);
  }
  if (parts.split.Schur && parts.split.physics)
    throw InputError("The PETSc options -dualmesh_schur and -dualmesh_split choose two splits; "
                     "give one of them.");
  return parts;
}

bool
prefer_direct_solver(int dimension, long long unknowns)
{
  // A direct factorisation of a sparse matrix from a mesh fills in: with a good ordering the
  // factors of a two-dimensional problem hold O(n log n) entries, but those of a three-dimensional
  // problem hold O(n^{4/3}) and cost O(n^2) operations (George, SIAM J. Numer. Anal. 10 (1973)
  // 345-363; Lipton, Rose and Tarjan, SIAM J. Numer. Anal. 16 (1979) 346-358).  So the direct
  // solver is kept where it is cheap: everywhere in one dimension, up to a large size in two and
  // only for small problems in three.
  if (dimension <= 1)
    return true;
  if (dimension == 2)
    return unknowns <= 100000;
  return unknowns <= 4000;
}

std::vector<linear_solve_attempt>
linear_solve_attempts(const linear_system_traits & system, const user_split & split)
{
  using kind = linear_solve_kind;
  // A split that the user chose must converge.
  if (split.Schur)
    return {{kind::Schur_split, *split.Schur, true}};
  if (split.physics)
    return {{kind::physics_blocks, SchurApproximation::LeastSquaresCommutator, true}};
  // Options of the user choose the solver themselves.
  if (!system.user_options.empty())
    return {{kind::user_options}};
  const linear_solve_attempt direct{kind::direct};
  if (prefer_direct_solver(system.dimension, system.global_unknowns))
    return {direct};
  // A large flow: flexible GMRES with the least squares commutator, which costs one
  // factorisation of the momentum block instead of one of the whole system.  Multigrid and an
  // incomplete factorisation of the whole saddle point system fail, since a Taylor-Hood system
  // has a zero pressure block.
  if (system.flow)
    return {{kind::Schur_split, SchurApproximation::LeastSquaresCommutator, false}, direct};
  if (system.mixed_order)
    return {direct};
  // A solid takes smoothed aggregation with its rigid body modes, a solid coupled to other physics
  // the split by physics, and any other system BoomerAMG, which scales with the processes.
  if (system.displacement_variables == 0)
    return {{kind::BoomerAMG, SchurApproximation::LeastSquaresCommutator, false}, direct};
  if (system.displacement_variables == system.variables)
    return {{kind::GAMG, SchurApproximation::LeastSquaresCommutator, false}, direct};
  return {{kind::physics_blocks, SchurApproximation::LeastSquaresCommutator, false}, direct};
}

} // namespace dualmesh
