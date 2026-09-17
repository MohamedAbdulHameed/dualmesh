// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Materials compute named properties at integration points (for example a
// conductivity, a stress tensor, or beam stiffness coefficients).  Kernels and
// boundary conditions look these properties up by name, which decouples the
// physics (kernels) from the constitutive model (materials), as in MOOSE.
#pragma once

#include "dualmesh/base/Object.h"
#include "dualmesh/base/QpContext.h"

#include <map>
#include <string>
#include <vector>

namespace dualmesh
{

/// Name -> (offset, number of components) of material property storage.
class MaterialPropertyRegistry
{
public:
  /// Declare a property; returns its id (storage offset).  Declaring the same
  /// name twice with the same size returns the existing id.
  int declare(const std::string & name, int components);
  /// Look up a property; throws a helpful error if it does not exist.
  int id(const std::string & name) const;
  int components(const std::string & name) const;
  bool has(const std::string & name) const { return _props.count(name) > 0; }
  int size() const { return _size; }
  const std::map<std::string, std::pair<int, int>> & all() const { return _props; }

private:
  std::map<std::string, std::pair<int, int>> _props;
  int _size = 0;
};

class Material : public Object
{
public:
  explicit Material(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;
  /// Declare the properties this material provides.
  virtual void declareProperties(MaterialPropertyRegistry & registry) = 0;
  /// Compute the declared properties at an integration point.
  virtual void computeProperties(QpContext & ctx) const = 0;

  const std::set<int> & blocks() const { return _blocks; }
  bool activeOnBlock(int b) const { return _blocks.empty() || _blocks.count(b); }

protected:
  int coupledVariable(Problem & problem, const std::string & param) const;
  std::set<int> _blocks;
};

} // namespace dualmesh
