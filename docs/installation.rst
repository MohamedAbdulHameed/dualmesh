Installation
============

You build dualmesh from its source.  The build compiles a C++17
library and its Python extension, and it links them to MPI, the Message Passing
Interface, and to PETSc, the Portable, Extensible Toolkit for Scientific
Computation.  The code builds on Linux and macOS.  Windows is not supported.

Requirements
------------

* A C++17 compiler.
* CMake 3.18 or newer.
* Python 3.11 or newer.
* An MPI library, such as Open MPI.
* PETSc with hypre and MUMPS, built on the same MPI library.  The build finds
  PETSc through pkg-config.
* pkg-config.
* Eigen 3.3 or newer.  Without an installed Eigen, the build downloads
  Eigen 3.4.0.

``pip`` installs the build tools, scikit-build-core 0.8 or newer and pybind11
2.11 or newer, and the packages the code needs at run time, NumPy 1.20 or newer
and PyYAML 6.0 or newer.

Install MPI, PETSc and Eigen
----------------------------

On Debian or Ubuntu, install the system packages:

.. code-block:: console

   sudo apt-get install -y libopenmpi-dev openmpi-bin libpetsc-real-dev pkg-config libeigen3-dev

On macOS, install them with Homebrew:

.. code-block:: console

   brew install open-mpi petsc pkg-config eigen

To use a PETSc that you built from source, add its pkg-config folder to the
search path of pkg-config:

.. code-block:: console

   export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH

Install the package
-------------------

1. Clone the repository:

   .. code-block:: console

      git clone https://github.com/MohamedAbdulHameed/dualmesh.git
      cd dualmesh

2. Build and install the package with all its optional dependencies:

   .. code-block:: console

      pip install ".[all]"

3. Check that the extension loads and finds PETSc:

   .. code-block:: console

      python -c "import dualmesh; print(dualmesh.petsc_version())"
      dualmesh --version
      dualmesh list --module heat_conduction

   The first command prints the version of PETSc, the second the version of
   dualmesh, and the third the objects of the heat transfer
   module.

The optional dependencies
-------------------------

Each group of optional dependencies is an extra of ``pip install``, as in
``pip install ".[mesh,plot]"``.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Extra
     - Packages
     - Use
   * - ``mesh``
     - meshio 5.0 or newer
     - Read and write mesh files.
   * - ``plot``
     - matplotlib 3.5 or newer
     - Plot the results of the examples.
   * - ``mms``
     - SymPy 1.9 or newer
     - Derive the sources of manufactured solutions.
   * - ``linalg``
     - SciPy 1.7 or newer
     - Solve eigenvalue problems with
       :meth:`dualmesh.Problem.solve_eigenvalue` and
       :meth:`dualmesh.Problem.solve_buckling`, and get the sparse Jacobian of
       :meth:`dualmesh.Problem.linear_system`.
   * - ``test``
     - pytest 7.0 or newer, meshio, SymPy, SciPy and PyYAML
     - Run the test suite.
   * - ``docs``
     - Sphinx 7.0 or newer, furo, myst-parser and sphinx-copybutton
     - Build this documentation.
   * - ``all``
     - meshio, matplotlib, SymPy, SciPy and PyYAML
     - Every run-time extra.

Build options
-------------

The CMake options of the build are:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Effect
   * - ``DUALMESH_BUILD_PYTHON``
     - ``ON``
     - Builds the Python extension module.
   * - ``DUALMESH_BUILD_TESTS``
     - ``ON``
     - Builds the C++ tests.  ``pip install`` sets it to ``OFF``.
   * - ``DUALMESH_ENABLE_OPENMP``
     - ``ON``
     - Threads the assembly with OpenMP.  Without OpenMP, the assembly runs on
       one thread.
   * - ``DUALMESH_ENABLE_GSLIB``
     - ``ON``
     - Downloads gslib 1.0.9, the gather-scatter library of Nek5000, checks it
       against its SHA-256 sum and builds it with ``make`` and the MPI C
       compiler.  gslib provides ``gather_scatter="gslib"`` of
       :class:`dualmesh.Problem`.
   * - ``DUALMESH_MAX_AD_DERIVATIVES``
     - ``96``
     - The largest number of derivatives that automatic differentiation seeds
       in one element, one for each node and field of the element.

Build in the source tree
------------------------

To work on the code, build the library, the C++ tests and the extension in a
``build`` folder:

1. Install the Python packages that the build needs:

   .. code-block:: console

      pip install pybind11 numpy pyyaml

2. Configure and build:

   .. code-block:: console

      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
      cmake --build build -j

   The build writes the extension module into ``python/dualmesh``.

3. Use the package from the source tree:

   .. code-block:: console

      export PYTHONPATH=$PWD/python

:doc:`developing` gives the commands that run the tests.
