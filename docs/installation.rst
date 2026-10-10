Installation
============

Requirements
------------

* a C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
* CMake 3.18 or newer
* Python 3.9 or newer
* `Eigen <https://eigen.tuxfamily.org>`_ 3.3 or newer, which is found
  automatically if installed and is otherwise downloaded during the build.  On
  macOS it is installed with ``brew install eigen cmake``, and on Debian or
  Ubuntu with ``apt install libeigen3-dev cmake``
* `pybind11 <https://pybind11.readthedocs.io>`_, which is obtained automatically
  by the build

Optional at run time: `meshio <https://github.com/nschloe/meshio>`_ for reading
and writing mesh files, and
`matplotlib <https://matplotlib.org>`_ for the plotting in the examples.

From PyPI
---------

Binary wheels for Linux, macOS (Intel and Apple silicon) and Windows, for
Python 3.9 to 3.13, are published on PyPI under the distribution name
``dualmesh-multiphysics``.  The name only differs on PyPI: the package is
imported as ``dualmesh`` and the command-line program is ``dualmesh``.

.. code-block:: console

   pip install dualmesh-multiphysics
   pip install "dualmesh-multiphysics[all]"   # meshio, matplotlib, SymPy, SciPy
   python -c "import dualmesh; print(dualmesh.__version__)"
   dualmesh --version

The wheels are built without MPI and PETSc.  To use either, build from source
as described below.

From source
-----------

.. code-block:: console

   git clone https://github.com/MohamedAbdulHameed/dualmesh.git
   cd dualmesh
   pip install .[all]

The build uses `scikit-build-core <https://scikit-build-core.readthedocs.io>`_,
so the C++ library and the Python extension are compiled by ``pip`` in one
step.  To work on the code, install it in editable mode:

.. code-block:: console

   pip install -e .[all,test]

Building the C++ library on its own
-----------------------------------

.. code-block:: console

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ctest --test-dir build --output-on-failure

This also builds the extension module into ``python/dualmesh``, so setting
``PYTHONPATH=$PWD/python`` is enough to use the package from the source tree
without installing it.

Optional: MPI and PETSc
-----------------------

The distributed solver needs MPI and PETSc: it communicates through PETSc's
star forests and solves with PETSc's Krylov methods and preconditioners, so a
build with MPI always uses PETSc.  A serial build may use PETSc too, for
``linear_solver="petsc"``.  With conda the packages are ``openmpi`` and
``petsc`` from conda-forge (PETSc there includes hypre and MUMPS); on Debian or
Ubuntu they are ``libopenmpi-dev`` and ``libpetsc-real-dev``.  PETSc is located
with ``pkg-config``, so a PETSc built from source is found by adding
``$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig`` to ``PKG_CONFIG_PATH``.  PETSc and
dualmesh must use the same MPI.  conda-forge has no PETSc for Windows, so a
distributed build on Windows runs under WSL.  A build with MPI also downloads
gslib, the gather-scatter library of Nek5000 (release 1.0.9, checked against
its SHA-256 sum), and builds it with ``make`` and the MPI C compiler, so that
``gather_scatter="gslib"`` is available next to ``gather_scatter="petsc"``;
``-DDUALMESH_ENABLE_GSLIB=OFF`` leaves it out.

.. code-block:: console

   pip install -C cmake.define.DUALMESH_ENABLE_MPI=ON .[all]
   python -c "import dualmesh as dm; print(dm.have_mpi(), dm.have_petsc(), dm.petsc_version())"

or, for the C++ library alone, ``-DDUALMESH_ENABLE_MPI=ON`` on the ``cmake``
command line.  A serial build with PETSc adds ``-DDUALMESH_ENABLE_PETSC=ON``.

Running the tests
-----------------

.. code-block:: console

   ctest --test-dir build --output-on-failure   # C++ unit tests
   pytest                                       # verification against the book

The Python suite reproduces the published dual mesh results of every worked
example of the book that has tabulated values (see :doc:`verification`).

Checking the installation
-------------------------

.. code-block:: console

   python -c "import dualmesh as dm; print(dm.__version__, len(dm.registered_types()))"
   dualmesh list --module heat_transfer
