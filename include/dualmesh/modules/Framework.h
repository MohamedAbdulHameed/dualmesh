// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"

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

class NeumannBC : public IntegratedBC
{
public:
  explicit NeumannBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

private:
  FunctionPtr _q;
};

class RobinBC : public IntegratedBC
{
public:
  explicit RobinBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

protected:
  FunctionPtr _h, _uinf, _q;
};

class GenericConstantMaterial : public Material
{
public:
  explicit GenericConstantMaterial(const InputParameters & p);
  static InputParameters validParams();
  void declareProperties(MaterialPropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

private:
  std::vector<std::string> _names;
  std::vector<double> _values;
  std::vector<int> _ids;
};

class GenericFunctionMaterial : public Material
{
public:
  explicit GenericFunctionMaterial(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  void declareProperties(MaterialPropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

private:
  std::vector<std::string> _names, _fnames;
  std::vector<FunctionPtr> _f;
  std::vector<int> _ids;
};

} // namespace dualmesh
