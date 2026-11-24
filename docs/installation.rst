Installation
============

Requirements
------------

* a C++17 compiler (GCC 9+ or Clang 10+) on Linux or macOS.  On Windows,
  dualmesh-multiphysics runs under WSL, the Windows Subsystem for Linux.
* CMake 3.18 or newer
* Python 3.11 or newer
* MPI and `PETSc <https://petsc.org>`_ 3.21 or newer, built with hypre and
  MUMPS.  The section `MPI and PETSc`_ gives the packages.
* `Eigen <https://eigen.tuxfamily.org>`_ 3.3 or newer, which is found
  automatically if installed and is otherwise downloaded during the build.  On
  macOS it is installed with ``brew install eigen cmake``, and on Debian or
  Ubuntu with ``apt install libeigen3-dev cmake``
* `pybind11 <https://pybind11.readthedocs.io>`_, which is obtained automatically
  by the build

Optional at run time: `meshio <https://github.com/nschloe/meshio>`_ for reading
and writing mesh files, and
`matplotlib <https://matplotlib.org>`_ for the plotting in the examples.

From source
-----------

.. code-block:: console

   git clone https://github.com/MohamedAbdulHameed/dualmesh-multiphysics.git
   cd dualmesh-multiphysics
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

MPI and PETSc
-------------

PETSc solves every linear system, in a serial problem and in a distributed
one, and a distributed problem communicates through PETSc's star forests over
MPI.  With conda, install ``openmpi`` and ``petsc`` from conda-forge, whose
PETSc includes hypre and MUMPS:

.. code-block:: console

   conda install -c conda-forge openmpi petsc

On Debian or Ubuntu the packages are ``libopenmpi-dev`` and
``libpetsc-real-dev``.  The build finds PETSc with ``pkg-config``.  To use a
PETSc built from source, add ``$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig`` to
``PKG_CONFIG_PATH``.  PETSc and dualmesh-multiphysics must use the same MPI.

The build also downloads gslib, the gather-scatter library of Nek5000
(release 1.0.9, checked against its SHA-256 sum), and builds it with ``make``
and the MPI C compiler.  gslib provides ``gather_scatter="gslib"`` next to
``gather_scatter="petsc"``.  ``-DDUALMESH_ENABLE_GSLIB=OFF`` on the ``cmake``
command line leaves it out.

To check which PETSc the extension uses, print its version:

.. code-block:: console

   python -c "import dualmesh; print(dualmesh.petsc_version())"

Running the tests
-----------------

.. code-block:: console

   ctest --test-dir build --output-on-failure   # C++ unit tests
   pytest                                       # the verification suite

The Python suite compares every computed value with an exact solution, a
published value or an independent code (see :doc:`verification`).

Checking the installation
-------------------------

.. code-block:: console

   python -c "import dualmesh; print(dualmesh.__version__, len(dualmesh.registered_types()))"
   dualmesh list --module heat_transfer
