// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Bindings of the solver options and results, the projection time integration of a flow, and
// the problem with its solves and its post-processing.
#include "bindings.h"
#include "dualmesh/base/Checkpoint.h"
#include "dualmesh/core/ParsedFunction.h"
#include "dualmesh/linalg/SaddlePointBlocks.h"
#include "dualmesh/modules/Projection.h"
#include "dualmesh/modules/SolidMechanics.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace dualmesh::bindings
{

namespace
{

/// A compressed sparse column matrix as the arrays (values, row indices, column pointers) of
/// scipy.sparse.csc_matrix.
std::tuple<py::array_t<double>, py::array_t<int>, py::array_t<int>>
compressedArrays(const SparseMatrix & J)
{
  return {toArray(J.nonZeros(), J.valuePtr()),
          toArray(J.nonZeros(), J.innerIndexPtr()),
          toArray(J.outerSize() + 1, J.outerIndexPtr())};
}

/// Matrix entries as the arrays (rows, columns, values).
py::tuple
tripletArrays(const std::vector<Eigen::Triplet<double>> & entries)
{
  std::vector<Index> rows, cols;
  std::vector<double> values;
  for (const auto & e : entries)
  {
    rows.push_back(e.row());
    cols.push_back(e.col());
    values.push_back(e.value());
  }
  return py::make_tuple(toArray(rows), toArray(cols), toArray(values));
}

using ProblemClass = py::class_<Problem>;

/// The solver options, the transient options and the results of a solve.
void
bindSolverOptions(py::module_ & m)
{
  py::class_<SolverOptions>(m, "SolverOptions")
      .def(py::init<>())
      .def_readwrite("nonlinear_solver", &SolverOptions::nonlinear_solver)
      .def_readwrite("max_iterations", &SolverOptions::max_iterations)
      .def_readwrite("relative_tolerance", &SolverOptions::relative_tolerance)
      .def_readwrite("relaxation", &SolverOptions::relaxation)
      .def_readwrite("line_search", &SolverOptions::line_search)
      .def_readwrite("max_line_search_steps", &SolverOptions::max_line_search_steps)
      .def_readwrite("petsc_options", &SolverOptions::petsc_options)
      .def_readwrite("linear_tolerance", &SolverOptions::linear_tolerance)
      .def_readwrite("verbose", &SolverOptions::verbose)
      .def_readwrite("error_on_divergence", &SolverOptions::error_on_divergence);

  py::class_<TransientOptions>(m, "TransientOptions")
      .def(py::init<>())
      .def_readwrite("start_time", &TransientOptions::start_time)
      .def_readwrite("end_time", &TransientOptions::end_time)
      .def_readwrite("dt", &TransientOptions::dt)
      .def_readwrite("theta", &TransientOptions::theta)
      .def_readwrite("second_order_scheme", &TransientOptions::second_order_scheme)
      .def_readwrite("spectral_radius", &TransientOptions::spectral_radius)
      .def_readwrite("output_times", &TransientOptions::output_times)
      .def_readwrite("time_stepper", &TransientOptions::time_stepper)
      .def_readwrite("courant_number", &TransientOptions::courant_number)
      .def_readwrite("dt_min", &TransientOptions::dt_min)
      .def_readwrite("dt_max", &TransientOptions::dt_max)
      .def_readwrite("growth_factor", &TransientOptions::growth_factor)
      .def_readwrite("cutback_factor", &TransientOptions::cutback_factor)
      .def_readwrite("error_tolerance", &TransientOptions::error_tolerance)
      .def_readwrite("optimal_iterations", &TransientOptions::optimal_iterations)
      .def_readwrite("iteration_window", &TransientOptions::iteration_window)
      .def_readwrite("max_rejected_steps", &TransientOptions::max_rejected_steps)
      .def_readwrite("max_steps", &TransientOptions::max_steps)
      .def_readwrite("checkpoint_file", &TransientOptions::checkpoint_file)
      .def_readwrite("checkpoint_interval", &TransientOptions::checkpoint_interval)
      .def_readwrite("restart_file", &TransientOptions::restart_file);
  m.def(
      "checkpoint_time",
      [](const std::string & file, bool distributed)
      {
        const auto point = checkpointTime(file, distributed);
        return py::make_tuple(point.time, point.dt, point.step);
      },
      py::arg("file"),
      py::arg("distributed"),
      "The time, the step size and the number of steps that a checkpoint records.");

  py::class_<IterationRecord>(m, "IterationRecord")
      .def_readonly("continuation_step", &IterationRecord::continuation_step)
      .def_readonly("data_fraction", &IterationRecord::data_fraction)
      .def_readonly("iteration", &IterationRecord::iteration)
      .def_readonly("residual_norm", &IterationRecord::residual_norm)
      .def_readonly("step_norm", &IterationRecord::step_norm);

  py::class_<SolveResult>(m, "SolveResult")
      .def_readonly("converged", &SolveResult::converged)
      .def_readonly("total_iterations", &SolveResult::total_iterations)
      .def_readonly("linear_iterations", &SolveResult::linear_iterations)
      .def_readonly("time_steps", &SolveResult::time_steps)
      .def_readonly("rejected_steps", &SolveResult::rejected_steps)
      .def_readonly("step_history", &SolveResult::step_history)
      .def_readonly("next_time_step", &SolveResult::next_time_step)
      .def_readonly("start_correction", &SolveResult::start_correction)
      .def_readonly("history", &SolveResult::history);

  m.def(
      "flush_output",
      []()
      {
        std::fflush(stdout);
        std::cout.flush();
      },
      "Write the progress lines that the library holds in its buffers, so that "
      "they come before the next lines of Python.");
}

/// The projection time integration of a flow.
void
bindProjection(py::module_ & m)
{
  py::class_<ProjectionSettings>(m, "ProjectionSettings")
      .def(py::init<>())
      .def_readwrite("order", &ProjectionSettings::order);
  py::class_<ProjectionSolver, std::shared_ptr<ProjectionSolver>>(m, "ProjectionSolver")
      .def_static(
          "coefficients",
          [](const std::vector<double> & times)
          {
            std::vector<double> beta, alpha;
            ProjectionSolver::coefficients(times, beta, alpha);
            return py::make_tuple(beta, alpha);
          },
          py::arg("times"));
  m.def(
      "attach_projection",
      [](Problem & problem, const ProjectionSettings & settings)
      {
        auto solver = std::make_shared<ProjectionSolver>(problem, settings);
        problem.setTimeIntegrator(solver);
        return solver;
      },
      py::arg("problem"),
      py::arg("settings"),
      "Advance the transient solves of @p problem with the projection time integration of its "
      "flow.");
  m.def(
      "attach_start_projection",
      [](Problem & problem,
         const std::vector<std::string> & velocities,
         const std::string & pressure)
      {
        problem.flowSystem().addStartProjection(
            [&problem, velocities, pressure](Vector & solution)
            {
              std::vector<int> components;
              for (const auto & name : velocities)
                components.push_back(problem.variableIndex(name));
              const int p = pressure.empty() ? -1 : problem.variableIndex(pressure);
              return projectInitialVelocity(problem, components, p, solution);
            });
      },
      py::arg("problem"),
      py::arg("velocities"),
      py::arg("pressure"),
      "Start the transient runs of @p problem from the projection of the initial velocity of "
      "the flow of @p velocities onto a field without divergence (an empty @p pressure for a "
      "flow without a pressure variable).");
}

/// The operators of a problem at its current solution, for the tests: the blocks of a flow
/// preconditioner, the steady linear system and the operator of an eigenvalue study.
void
defineOperators(ProblemClass & cls)
{
  cls.def(
         "_flow_operators",
         [](Problem & p, const std::string & preconditioner)
         {
           // The operators of a flow preconditioner (SaddlePointBlocks.h) at the current
           // solution, for the tests: each operator as (rows, columns, values) in the problem's
           // numbering, and the lumped masses per unknown.
           const auto approximation = schurApproximation(preconditioner);
           if (!approximation)
             throw InputError("_flow_operators takes pressure_mass, pcd or lsc.");
           SaddlePointBlocks blocks;
           {
             py::gil_scoped_release release;
             Vector R;
             SparseMatrix J;
             p.linearSystem(R, J);
             blocks = p.flowSystem().saddlePointBlocks(p, J, *approximation);
           }
           std::vector<int> constrained(blocks.constrained.begin(), blocks.constrained.end());
           py::dict out;
           out["pressure_dofs"] = toArray(blocks.pressure_dofs);
           out["other_dofs"] = toArray(blocks.other_dofs);
           out["constrained"] = toArray(constrained);
           out["lumped_mass"] = toArray(blocks.lumped_mass);
           out["sign"] = blocks.sign;
           out["coupling"] = blocks.coupling;
           out["refusal"] = blocks.refusal;
           out["mass"] = tripletArrays(blocks.mass);
           out["stabilization"] = tripletArrays(blocks.stabilization);
           out["commutator_stabilization"] = tripletArrays(blocks.commutator_stabilization);
           out["convection_diffusion"] = tripletArrays(blocks.convection_diffusion);
           return out;
         },
         py::arg("preconditioner"))
      .def("_linear_system",
           [](Problem & p)
           {
             // The steady residual and its Jacobian at the current solution, as one Newton step
             // of solve_steady sees them (Problem::linearSystem).
             Vector R;
             SparseMatrix J;
             {
               py::gil_scoped_release release;
               p.linearSystem(R, J);
             }
             const auto [values, indices, indptr] = compressedArrays(J);
             return py::make_tuple(toArray(R), values, indices, indptr);
           })
      .def(
          "_operator",
          [](Problem & p, bool steady_terms, bool time_kernels, double eigenvalue, int order)
          {
            // The operator of an eigenvalue study (Problem::operatorOf) at the current solution,
            // with the equations of the dependent degrees of freedom folded into those they
            // depend on, and the mask of the prescribed and dependent degrees of freedom, whose
            // rows and columns the caller removes.
            SparseMatrix J;
            std::vector<char> constrained;
            {
              py::gil_scoped_release release;
              p.initialize();
              Problem::OperatorOptions options;
              options.steady_terms = steady_terms;
              options.time_kernels = time_kernels;
              options.at_solution = true;
              options.eigenvalue = eigenvalue;
              options.order = order;
              J = p.operatorOf(options);
              constrained = p.constrainedDofs();
            }
            const auto [values, indices, indptr] = compressedArrays(J);
            py::array_t<bool> mask(J.rows());
            for (Index i = 0; i < J.rows(); ++i)
              mask.mutable_at(i) = constrained[i] != 0;
            return py::make_tuple(values, indices, indptr, mask);
          },
          py::arg("steady_terms"),
          py::arg("time_kernels"),
          py::arg("eigenvalue"),
          py::arg("order") = 1);
}

/// The variables, functions, fields and objects of a problem, and its solution.
void
defineVariablesAndObjects(ProblemClass & cls)
{
  cls.def("second_order_variables",
          [](Problem & p)
          {
            p.initialize();
            std::vector<std::string> names;
            const auto & flags = p.secondOrderVariables();
            for (std::size_t v = 0; v < flags.size(); ++v)
              if (flags[v])
                names.push_back(p.variable(static_cast<int>(v)).name);
            return names;
          })
      .def("set_initial_velocity",
           [](Problem & p, const std::vector<double> & v) {
             p.setInitialVelocity(Eigen::Map<const Vector>(v.data(), static_cast<Index>(v.size())));
           })
      .def("velocity",
           [](const Problem & p)
           {
             const Vector v = p.velocity();
             return std::vector<double>(v.data(), v.data() + v.size());
           })
      .def("set_num_threads", &Problem::setNumThreads, py::arg("num_threads"))
      .def("num_threads", &Problem::numThreads)
      .def("effective_threads", &Problem::effectiveThreads)
      .def("thread_safe", &Problem::threadSafe)
      .def("extend_variable_blocks",
           &Problem::extendVariableBlocks,
           py::arg("name"),
           py::arg("blocks"),
           "Let a variable also live on these blocks (an empty list: everywhere).")
      .def(
          "add_variable",
          [](Problem & p,
             const std::string & name,
             const std::vector<std::string> & blocks,
             const py::object & initial_condition,
             const std::string & order)
          {
            // A text is the name of a registered function or an expression of (x, y, z, t).
            FunctionPtr ic;
            if (py::isinstance<py::str>(initial_condition))
            {
              const auto text = py::cast<std::string>(initial_condition);
              ic = p.hasFunction(text) ? p.function(text) : std::make_shared<ParsedFunction>(text);
            }
            else if (!initial_condition.is_none())
              ic = toFunction(initial_condition);
            VariableOrder o;
            if (order == "mesh")
              o = VariableOrder::Mesh;
            else if (order == "first")
              o = VariableOrder::First;
            else
              throw InputError("Unknown variable order '" + order +
                               "'. Use 'mesh' (the order of the elements) or 'first' (linear "
                               "on every element, the pressure of a Taylor-Hood element).");
            return p.addVariable(name, blocks, ic, 1.0, o);
          },
          py::arg("name"),
          py::arg("blocks") = std::vector<std::string>{},
          py::arg("initial_condition") = py::none(),
          py::arg("order") = "mesh")
      .def(
          "variable_order",
          [](const Problem & p, const std::string & name)
          {
            return p.variable(p.variableIndex(name)).order == VariableOrder::First
                       ? std::string("first")
                       : std::string("mesh");
          },
          py::arg("name"))
      .def("set_variable_unit", &Problem::setVariableUnit, py::arg("name"), py::arg("unit"))
      .def(
          "variable_unit",
          [](const Problem & p, const std::string & name)
          { return p.variable(p.variableIndex(name)).unit; },
          py::arg("name"))
      .def("add_function",
           [](Problem & p, const std::string & name, const py::object & f)
           { p.addFunction(name, toFunction(f)); })
      .def("set_element_field",
           &Problem::setElementField,
           py::arg("name"),
           py::arg("values"),
           "Set a named field with one value per element, read by the property objects.")
      .def(
          "element_field",
          [](Problem & p, const std::string & name) { return p.elementField(name); },
          py::arg("name"))
      .def(
          "add_object",
          [](Problem & p, const std::string & type, const py::kwargs & kwargs) {
            return p.addObject(
                type, "", paramsFromKwargs(Factory::instance().validParams(type), kwargs));
          },
          py::arg("type"))
      .def("add_kernel", &Problem::addKernel)
      .def("add_integrated_bc", &Problem::addIntegratedBC)
      .def("add_nodal_bc", &Problem::addNodalBC)
      .def("add_nodal_load", &Problem::addNodalLoad)
      .def("add_property", &Problem::addProperty)
      .def("initialize", &Problem::initialize)
      .def(
          "near_nullspace",
          [](Problem & p)
          {
            p.initialize();
            const Eigen::MatrixXd B = p.solidSystem().nearNullspace(p);
            std::vector<std::vector<double>> rows(
                static_cast<std::size_t>(B.rows()),
                std::vector<double>(static_cast<std::size_t>(B.cols())));
            for (Eigen::Index i = 0; i < B.rows(); ++i)
              for (Eigen::Index j = 0; j < B.cols(); ++j)
                rows[i][j] = B(i, j);
            return rows;
          },
          "The near null space of the algebraic multigrid preconditioner, one row per degree of "
          "freedom.")
      .def("boundary_area",
           &Problem::boundaryArea,
           py::arg("boundaries"),
           "The measure of the named side sets, with the coordinate factor of axisymmetric and "
           "spherical problems.")
      .def(
          "num_active_dofs",
          [](Problem & p)
          {
            p.initialize();
            Index n = 0;
            for (char a : p.activeDofs())
              n += a ? 1 : 0;
            return n;
          },
          "The number of degrees of freedom that carry an equation: every variable at every node "
          "where a kernel acts on it, and for a first-order variable on a quadratic mesh only the "
          "corner nodes.")
      .def_property_readonly("mesh", &Problem::meshPointer)
      .def_property_readonly("num_variables", &Problem::numVariables)
      .def("variable_index", &Problem::variableIndex)
      .def("variable_name", [](const Problem & p, int i) { return p.variable(i).name; })
      .def("property_id",
           [](Problem & p, const std::string & n) { return p.propertyRegistry().id(n); })
      .def_property("time", &Problem::time, &Problem::setTime)
      .def("values", [](const Problem & p, const std::string & v) { return toArray(p.values(v)); })
      .def("set_values",
           [](Problem & p, const std::string & v, const std::vector<double> & x)
           { p.setValues(v, x); })
      .def("solution", [](const Problem & p) { return toArray(p.solution()); })
      .def(
          "set_solution",
          [](Problem & p, const std::vector<double> & x)
          {
            if (static_cast<Index>(x.size()) != p.solution().size())
              throw InputError("set_solution: " + std::to_string(x.size()) + " values for " +
                               std::to_string(p.solution().size()) + " unknowns.");
            for (std::size_t i = 0; i < x.size(); ++i)
              p.solution()[static_cast<Index>(i)] = x[i];
          },
          py::arg("values"))
      .def("apply_initial_conditions", &Problem::applyInitialConditions);
}

/// The steady and transient solves, the parameters and sensitivities, and the callbacks.
void
defineSolves(ProblemClass & cls)
{
  cls.def("solve_steady", &Problem::solveSteady, py::arg("options") = SolverOptions())
      .def("add_parameter",
           &Problem::addParameter,
           py::arg("name"),
           py::arg("value"),
           "Register a named settable constant: a parameter that sensitivities() differentiates "
           "with respect to. Give its name wherever a coefficient, a source or a boundary value "
           "takes a function.")
      .def("set_parameter",
           &Problem::setParameter,
           py::arg("name"),
           py::arg("value"),
           "Assign a registered parameter; solve() gives the solution of the new value.")
      .def("parameter_value", &Problem::parameterValue, py::arg("name"))
      .def(
          "objective_value",
          [](Problem & p, const std::string & variable, const std::string & boundary)
          {
            Problem::Objective objective{variable, boundary};
            return p.objectiveValue(objective);
          },
          py::arg("variable"),
          py::arg("boundary") = "",
          "The integral of a variable over the whole domain (an empty boundary) or over one "
          "side set, with the coordinate factor: the value of the objective of sensitivities().")
      .def(
          "sensitivities",
          [](Problem & p,
             const std::string & variable,
             const std::string & boundary,
             const std::vector<std::string> & parameters,
             const SolverOptions & options)
          {
            py::gil_scoped_release release;
            Problem::Objective objective{variable, boundary};
            return p.sensitivities(objective, parameters, options);
          },
          py::arg("variable"),
          py::arg("boundary") = "",
          py::arg("parameters"),
          py::arg("options") = SolverOptions(),
          "d(objective)/d(parameter) of every named settable constant, by one adjoint solve "
          "with the exact Jacobian of the converged solution. The objective is the integral of "
          "the variable over the whole domain (an empty boundary) or over one side set. The "
          "problem must hold a converged solution (solve_steady leaves one).")
      .def("solve_transient",
           &Problem::solveTransient,
           py::arg("transient"),
           py::arg("options") = SolverOptions())
      .def(
          "project_initial_velocity",
          [](Problem & p) { return p.flowSystem().projectInitialVelocity(p); },
          "Project the velocity of every flow onto a field without divergence that keeps the "
          "prescribed values, as a transient run does before its first step, and return the "
          "largest change of the velocity over the largest speed, or None for a problem "
          "without a flow. Collective in a distributed problem.")
      .def("set_time_step_callback",
           [](Problem & p, const py::function & f)
           {
             p.setTimeStepCallback(
                 [f](double t, Problem & pb)
                 {
                   py::gil_scoped_acquire gil;
                   f(t, py::cast(&pb, py::return_value_policy::reference));
                 });
           })
      .def("set_output_callback",
           [](Problem & p, const py::object & f)
           {
             if (f.is_none())
             {
               p.setOutputCallback(nullptr);
               return;
             }
             p.setOutputCallback(
                 [f](int index)
                 {
                   py::gil_scoped_acquire gil;
                   f(index);
                 });
           })
      .def("set_checkpoint_callback",
           [](Problem & p, const py::object & f)
           {
             if (f.is_none())
             {
               p.setCheckpointCallback(nullptr);
               return;
             }
             p.setCheckpointCallback(
                 [f]()
                 {
                   py::gil_scoped_acquire gil;
                   f();
                 });
           });
}

/// The post-processing: outflows, samples, integrals, error norms, crack tip integrals and
/// output.
void
definePostprocessing(ProblemClass & cls)
{
  cls.def(
         "boundary_outflow",
         [](const Problem & p, const std::string & variable, const std::string & boundary)
         { return p.boundaryOutflow(variable, boundary); },
         py::arg("variable"),
         py::arg("boundary"))
      .def("total_outflow", &Problem::totalOutflow)
      .def("sample", &Problem::sample)
      .def("sample_gradient", &Problem::sampleGradient)
      .def("gradient_at_centroids", &Problem::gradientAtCentroids)
      .def("error_indicator",
           &Problem::errorIndicator,
           py::arg("variable"),
           "One error indicator per element, from gradient recovery.")
      .def("property_at_centroids", &Problem::propertyAtCentroids)
      .def("property_at_points", &Problem::propertyAtPoints)
      .def("kernel_flux_at_centroids", &Problem::kernelFluxAtCentroids)
      .def("section_stress_at_centroids",
           [](Problem & p, const std::string & kernel, double z)
           { return sectionStressAtCentroids(p, kernel, z); })
      .def("integrate", &Problem::integrate)
      .def("element_integrals", &Problem::elementIntegrals)
      .def("node_measures",
           &Problem::nodeMeasures,
           "The measure of each node as the nodal rule weighs it: that of its control domain "
           "(DMCDM) or the integral of its shape function (FEM), with the coordinate factor at "
           "the node.")
      .def(
          "error_norms",
          [](const Problem & p,
             const std::string & variable,
             const py::object & exact,
             const py::object & gradient,
             int quadrature_points)
          {
            const FunctionPtr u = toFunction(exact);
            std::array<FunctionPtr, 3> g;
            std::array<const Function *, 3> gp = {nullptr, nullptr, nullptr};
            if (!gradient.is_none())
            {
              const auto list = py::cast<py::sequence>(gradient);
              for (std::size_t d = 0; d < 3; ++d)
              {
                g[d] =
                    d < list.size() ? toFunction(list[d]) : std::make_shared<ConstantFunction>(0.0);
                gp[d] = g[d].get();
              }
            }
            Problem::ErrorNorms norms;
            {
              // Python callables take the lock back themselves; everything
              // else runs without it, on several threads.
              py::gil_scoped_release release;
              norms = p.errorNorms(variable, *u, gp, quadrature_points);
            }
            return py::make_tuple(norms.l2, norms.h1_seminorm);
          },
          py::arg("variable"),
          py::arg("exact"),
          py::arg("exact_gradient") = py::none(),
          py::arg("quadrature_points") = 0,
          "The L2 norm and the H1 seminorm of the error against an exact solution.")
      .def("kernel_boundary_flux", &Problem::boundaryFluxIntegral)
      .def("normal_integral", &Problem::normalIntegral)
      .def("crack_tip_integrals",
           [](Problem & p,
              const std::vector<std::string> & displacements,
              const std::vector<double> & tip,
              double angle,
              double inner,
              double outer,
              double E,
              double nu,
              bool plane_strain)
           {
             return crackTipIntegrals(p,
                                      displacements,
                                      Point{tip.at(0), tip.at(1), 0.0},
                                      angle,
                                      inner,
                                      outer,
                                      E,
                                      nu,
                                      plane_strain);
           })
      .def("_last_residual",
           [](const Problem & p)
           {
             const Vector & R = p.lastResidual();
             return toArray(R);
           })
      .def("write_vtu",
           &Problem::writeVTU,
           py::arg("filename"),
           py::arg("cell_properties") = std::vector<std::string>{},
           py::arg("fields") = std::vector<std::string>{})
      .def("summary", &Problem::summary)
      .def("object_names", &Problem::objectNames)
      .def("object", &Problem::object, py::arg("name"))
      .def("variable_names",
           [](const Problem & p)
           {
             std::vector<std::string> names;
             for (int v = 0; v < p.numVariables(); ++v)
               names.push_back(p.variable(v).name);
             return names;
           });
}

} // namespace

void
bindProblem(py::module_ & m)
{
  bindSolverOptions(m);
  bindProjection(m);
  ProblemClass cls(m, "Problem");
  cls.def(py::init(
              [](std::shared_ptr<Mesh> mesh,
                 const std::string & method,
                 const std::string & coordinates)
              {
                return std::make_unique<Problem>(
                    std::move(mesh), methodFromString(method), coordFromString(coordinates));
              }),
          py::arg("mesh"),
          py::arg("method"),
          py::arg("coordinates") = "cartesian")
      .def(
          "set_method",
          [](Problem & p, const std::string & s) { p.setMethod(methodFromString(s)); },
          py::arg("method"))
      .def("is_dual", &Problem::isDual);
  defineOperators(cls);
  defineVariablesAndObjects(cls);
  defineSolves(cls);
  definePostprocessing(cls);
}

} // namespace dualmesh::bindings
