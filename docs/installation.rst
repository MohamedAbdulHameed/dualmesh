Installation
============

Requirements
------------

* a C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
* CMake 3.18 or newer
* Python 3.9 or newer
* `Eigen <https://eigen.tuxfamily.org>`_ 3.3 or newer — found automatically if
  installed, otherwise downloaded during the build.  On macOS
  ``brew install eigen cmake`` is the quickest route; on Debian or Ubuntu,
  ``apt install libeigen3-dev cmake``
* `pybind11 <https://pybind11.readthedocs.io>`_ — pulled in by the build

Optional at run time: `meshio <https://github.com/nschloe/meshio>`_ for reading
and writing mesh files, `PyYAML <https://pyyaml.org>`_ for input files, and
`matplotlib <https://matplotlib.org>`_ for the plotting in the examples.

From source
-----------

.. code-block:: console

   git clone https://github.com/OWNER/dualmesh.git
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

Running the tests
-----------------

.. code-block:: console

   ctest --test-dir build --output-on-failure   # C++ unit tests
   pytest                                       # verification against the book

The Python suite reproduces the published dual mesh results of every worked
example of the book that has tabulated values; see :doc:`verification`.

Checking the installation
-------------------------

.. code-block:: console

   python -c "import dualmesh as dm; print(dm.__version__, len(dm.registered_types()))"
   dualmesh list --module heat_transfer
