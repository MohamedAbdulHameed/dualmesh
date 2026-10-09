// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Property.h"

namespace dualmesh
{

class Diffusion : public Kernel
{
public:
  explicit Diffusion(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;

protected:
  ADReal coefficient(const QpContext & ctx) const;
  FunctionPtr _k;
  std::vector<double> _poly;
  std::string _prop_name;
  int _prop = -1;
};

class AnisotropicDiffusion : public Kernel
{
public:
  explicit AnisotropicDiffusion(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;

private:
  std::vector<double> _entries;
  std::array<std::array<double, 3>, 3> _K;
};

class Reaction : public Kernel
{
public:
  explicit Reaction(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  FunctionPtr _c;
  double _p;
  std::string _prop_name;
  int _prop = -1;
};

class BodyForce : public Kernel
{
public:
  explicit BodyForce(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  FunctionPtr _f;
};

class TimeDerivative : public Kernel
{
public:
  explicit TimeDerivative(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  FunctionPtr _c;
};

class Advection : public Kernel
{
public:
  explicit Advection(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return _conservative; }
  bool hasSource() const override { return !_conservative; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;
  ADReal computeSource(const QpContext & ctx) const override;

private:
  bool _conservative = false;
  std::array<FunctionPtr, 3> _v;
};

class CoupledForce : public Kernel
{
public:
  explicit CoupledForce(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  int _v = -1;
  FunctionPtr _c;
  std::string _prop_name;
  int _prop = -1;
};

class DirichletBC : public NodalBC
{
public:
  explicit DirichletBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  double computeValue(const Point & x, double t) const override;

private:
  FunctionPtr _g;
};

/// Prescribed value at the one entity nearest to a point: the usual way to fix
/// the level of a pressure that is otherwise determined only up to a constant.
class PointDirichletBC : public NodalBC
{
public:
  explicit PointDirichletBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  double computeValue(const Point & x, double t) const override;
  /// Distance from the requested point to the entity chosen on this process.
  double distance() const { return _distance; }
  /// Drop the entity (a distributed run keeps it only on the process whose
  /// entity is nearest to the point).
  void clear() { _nodes.clear(); }

private:
  FunctionPtr _g;
  double _distance = 0.0;
};

class NeumannBC : public IntegratedBC
{
public:
  explicit NeumannBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

private:
  FieldCoefficient _q;
};

class RobinBC : public IntegratedBC
{
public:
  explicit RobinBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

protected:
  FieldCoefficient _h, _uinf, _q;
};

class ConstantProperty : public Property
{
public:
  explicit ConstantProperty(const InputParameters & p);
  static InputParameters validParams();
  void declareProperties(PropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

private:
  std::vector<std::string> _names;
  std::vector<double> _values;
  std::vector<int> _ids;
};

class FunctionProperty : public Property
{
public:
  explicit FunctionProperty(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  void declareProperties(PropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

private:
  std::vector<std::string> _names, _fnames;
  std::vector<FunctionPtr> _f;
  std::vector<int> _ids;
};

} // namespace dualmesh
