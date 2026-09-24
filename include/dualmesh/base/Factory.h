// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Object registry and factory (MOOSE "registerMooseObject" equivalent).
#pragma once

#include "dualmesh/base/Object.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

enum class ObjectCategory
{
  Kernel,
  BoundaryCondition,
  NodalBC,
  NodalLoad,
  Material
};

std::string categoryName(ObjectCategory c);

class Factory
{
public:
  using ParamsFunction = std::function<InputParameters()>;
  using BuildFunction = std::function<std::shared_ptr<Object>(const InputParameters &)>;

  static Factory & instance();

  template <typename T>
  void add(const std::string & type, ObjectCategory category, const std::string & module)
  {
    Entry e;
    e.category = category;
    e.module = module;
    e.params = [] { return T::validParams(); };
    e.build = [](const InputParameters & p) { return std::make_shared<T>(p); };
    _entries[type] = e;
  }

  bool has(const std::string & type) const { return _entries.count(type) > 0; }
  InputParameters validParams(const std::string & type) const;
  ObjectCategory category(const std::string & type) const;
  std::string module(const std::string & type) const;
  std::shared_ptr<Object>
  create(const std::string & type, const std::string & name, InputParameters params) const;
  std::vector<std::string> registeredTypes() const;

private:
  Factory();
  struct Entry
  {
    ObjectCategory category;
    std::string module;
    ParamsFunction params;
    BuildFunction build;
  };
  const Entry & entry(const std::string & type) const;
  std::map<std::string, Entry> _entries;
};

// Module registration functions (called once by the Factory constructor).
void registerFrameworkObjects(Factory & f);
void registerHeatTransferObjects(Factory & f);
/// Solid mechanics: continuum elasticity and, through
/// registerStructuralMemberObjects, the beams and plates.
void registerSolidMechanicsObjects(Factory & f);
void registerStructuralMemberObjects(Factory & f);
void registerFluidObjects(Factory & f);

} // namespace dualmesh
