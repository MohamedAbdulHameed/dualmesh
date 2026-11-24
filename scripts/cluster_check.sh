#!/bin/bash
#SBATCH --job-name=dualmesh_check
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --output=dualmesh_check_%j.out
#
# Builds and tests dualmesh on a cluster, and runs the FEniCSx benchmarks.
#
#   bash scripts/cluster_check.sh setup   # once, on a login node with internet
#   sbatch scripts/cluster_check.sh run   # or: bash scripts/cluster_check.sh run
#
# Submit from the top folder of the repository: Slurm runs a copy of this
# script, and the repository is then the folder of the submission.
#
# The setup creates two conda-forge environments under $HOME/envs. The run
# writes one log per step and a summary to cluster_logs/. Every step runs even
# when an earlier one fails, so that one run reports every failure.
set -o pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ENVS=${DUALMESH_ENVS:-$HOME/envs}
MAMBA=${MAMBA:-$HOME/.local/bin/micromamba}
LOGS=$ROOT/cluster_logs

setup() {
  if [ ! -x "$MAMBA" ]; then
    mkdir -p "$(dirname "$MAMBA")"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$(dirname "$MAMBA")" --strip-components=1 bin/micromamba
  fi
  "$MAMBA" create -y -p "$ENVS/dualmesh" -c conda-forge \
    python=3.12 petsc mpich mpi4py eigen cmake ninja pkg-config cxx-compiler \
    numpy scipy sympy meshio matplotlib pyyaml pytest ruff=0.16.10 gmsh python-gmsh
  "$MAMBA" create -y -p "$ENVS/fenicsx" -f "$ROOT/verification/fenicsx/environment.yml"
}

# Runs one step, logs its output and records its exit code in the summary.
step() {
  local name=$1
  shift
  echo "== $name"
  (cd "$ROOT" && "$@") >"$LOGS/$name.log" 2>&1
  local status=$?
  printf '%-28s %s\n' "$name" "$([ $status -eq 0 ] && echo passed || echo "FAILED ($status)")" >>"$LOGS/summary.txt"
}

run() {
  mkdir -p "$LOGS"
  : >"$LOGS/summary.txt"
  eval "$("$MAMBA" shell hook -s bash)"

  micromamba activate "$ENVS/dualmesh"
  export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
  step configure cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DDUALMESH_BUILD_PYTHON=OFF
  step build cmake --build build -j 4
  step cpp_tests ctest --test-dir build --output-on-failure
  for n in 1 2 3 4; do
    step "cpp_parallel_$n" mpiexec -n $n ./build/dualmesh_parallel_tests
  done
  step install pip install --no-build-isolation \
    ".[all,test]"
  step python_tests python -m pytest -q
  for n in 1 2 3 4; do
    step "python_parallel_$n" mpiexec -n $n python -m pytest -q -p no:cacheprovider tests/python/test_parallel.py
  done
  step ruff ruff check python tests examples verification
  step names python scripts/audit_names.py
  step name_rules python scripts/check_names.py tests/python examples verification
  for example in bus_bar cooling_fin transient_slab; do
    step "example_$example" bash -c "cd examples && python $example.py"
  done
  for benchmark in solids/run_Cook solids/run_NAFEMS_LE1; do
    step "benchmark_$(basename $benchmark)" python "verification/benchmarks/$benchmark.py" "$LOGS/figures"
  done
  # The adapted DFG 2D-1 on 1 and 4 processes, each holding the hierarchy and distributing the problems.
  for n in 1 4; do
    step "benchmark_dfg_adapted_$n" mpiexec -n $n python verification/benchmarks/fluids/run_dfg_adapted.py "$LOGS/figures"
  done
  micromamba deactivate

  micromamba activate "$ENVS/fenicsx"
  for benchmark in Kovasznay cavity dfg_steady Cook; do
    step "fenicsx_$benchmark" bash -c "cd verification/fenicsx && mpiexec -n 4 python $benchmark.py"
  done
  micromamba deactivate

  cat "$LOGS/summary.txt"
}

case ${1:-run} in
  setup) setup ;;
  run) run ;;
  *) echo "Usage: $0 setup | run" >&2; exit 2 ;;
esac
