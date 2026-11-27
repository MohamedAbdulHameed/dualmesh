// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Object registry and factory: every object type is registered by name, and created by name with validated parameters.
#pragma once

#include "dualmesh/base/FlowSystem.h"
#include "dualmesh/base/Object.h"
#include "dualmesh/base/SolidSystem.h"

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
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
  Property,
  Constraint
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
    if (!_entries.emplace(type, e).second)
      throw std::logic_error("dualmesh: the object type '" + type + "' is registered twice.");
  }

  bool has(const std::string & type) const { return _entries.count(type) > 0; }
  InputParameters validParams(const std::string & type) const;
  ObjectCategory category(const std::string & type) const;
  std::string module(const std::string & type) const;
  std::shared_ptr<Object>
  create(const std::string & type, const std::string & name, InputParameters params) const;
  std::vector<std::string> registeredTypes() const;

  /// The systems of the physics, which the modules register besides their objects: the flow
  /// system, of which every Problem makes its own, and the solid system, which every Problem
  /// shares (Problem::flowSystem, Problem::solidSystem).
  void setFlowSystem(std::function<std::unique_ptr<FlowSystem>()> make)
  {
    _make_flow_system = std::move(make);
  }
  void setSolidSystem(std::shared_ptr<const SolidSystem> solid)
  {
    _solid_system = std::move(solid);
  }
  /// A new flow system, or null when no module registered one.
  std::unique_ptr<FlowSystem> makeFlowSystem() const
  {
    return _make_flow_system ? _make_flow_system() : nullptr;
  }
  /// The solid system, or null when no module registered one.
  std::shared_ptr<const SolidSystem> solidSystem() const { return _solid_system; }

private:
  /// Empty: the modules register their objects (registerAllObjects of modules/Registry.h).
  Factory() = default;
  struct Entry
  {
    ObjectCategory category;
    std::string module;
    ParamsFunction params;
    BuildFunction build;
  };
  const Entry & entry(const std::string & type) const;
  std::map<std::string, Entry> _entries;
  std::function<std::unique_ptr<FlowSystem>()> _make_flow_system;
  std::shared_ptr<const SolidSystem> _solid_system;
};

} // namespace dualmesh
