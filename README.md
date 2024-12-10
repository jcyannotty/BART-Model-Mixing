# Open Bayesian Trees Project
This repository includes new developments with Bayesian Additive Regression Trees and extends the original OpenBT repository created by Matt Pratola (https://bitbucket.org/mpratola/openbt/src/master/).
Such extensions include Bayesian Model Mixing and Bayesian Calibration. 
All of the Bayesian Tree code is written in C++. User interfaces constructed in R and Python allow one to easily run the software.
The BART Model Mixing software has been implemented in the [Taweret](https://github.com/TaweretOrg/Taweret/tree/main) Python package in conjunction with the [BAND](https://bandframework.github.io/) collaboration.


# Installation
The heart of OpenBTMixing is a set of C++ tools that can be used directly via
the command line or indirectly through the Python and R packages, which wrap
them.  Typically these tools are built with an implementation of the Message
Passing Interface (MPI) such as [Open MPI](https://www.open-mpi.org) or
[MPICH](https://www.mpich.org) to enable distributed parallelization of
computations.  In particular, the Python wrapper package is always built with
MPI support.

The software and its distribution scheme have been developed to allow users to
use OpenBTMixing with the MPI installation of their choice.  For instance, it
can be built with MPI installations on leadership class platforms and clusters
that were installed by experts and optimized for their specific platform.  As a
result, however, the software is not distributed as prebuilt binaries or wheels,
but rather must be built for each case with the compiler suite and matching MPI
implementation provided by the user.

## Requirements
Before building and installing the bare C++ tools or the Python package, users
must provide
* a compiler suite that includes a C++ compiler that supports the C++11
  standard,
* an MPI installation that is compatible with the compiler suite,
* the [Meson build system](https://mesonbuild.com) and its prerequistes such as
  Python 3 and [ninja](https://ninja-build.org), and
* optionally the [Eigen software package](https://gitlab.com/libeigen/eigen).

The Meson build system is setup to automatically detect the compiler suite and
MPI installation to use.  If Eigen already exists in the system and Meson can
find it, then Meson will use it for the build.  Otherwise, Meson will
automatically obtain a copy of Eigen for internal use.

We presently test OpenBTMixing with both Open MPI and MPICH.  In addition, we
have successfully tested with the Intel MPI implementation and have used the
Python package with MPI implementations installed
* via package managers such as Ubuntu's Advanced Packaging Tool (`apt`) and
  [homebrew](https://brew.sh) on macOS;
* by experts on clusters and that are available as modules; and
* with `conda` from the prebuilt conda forge
  [openmpi](https://anaconda.org/conda-forge/openmpi) package.

## Meson installation
The Meson build system documentation suggests installing Meson via package
manager when possible.  Please refer to that documentation for detailed and
up-to-date installation information.

If Meson cannot be installed by package manager or the manager's version is too
old, the following is contrary to Meson suggestions but has been used
successfully to install Meson with Python into a dedicated virtual environment
as well as to install `meson` in the `PATH` for use without needing to activate
that virtual environment.
```
$ /path/to/target/python -m venv ~/local/venv/meson
$ . ~/local/venv/meson/bin/activate
$ which python
$ python -m pip install --upgrade pip
$ python -m pip install meson
$ python -m pip list
$ ln -s ~/local/venv/meson/bin/meson ~/local/bin
<add ~/local/bin to PATH if desired and appropriate>
$ deactivate
$ which meson
$ meson --version
```
Note that this `meson` virtual environment is for installing **just** the Meson
build system.  Attempts to install `openbtmixing` into this virtual environment
will likely fail with an error that the `mesonbuild` module cannot be found.

## Python package
The OpenBTMixing Python package is distributed on
[PyPI](https://pypi.org/project/openbtmixing/) as a source distribution that
contains the C++ code and files needed by Meson to build the dedicated,
standalone command line tools that the package will use.  The tools are built
and installed automatically by Meson as part of executing
```
python -m pip install openbtmixing
```
By default, `pip install` does not show any of Meson's progress.  Users and
developers interested in seeing how Meson satisifies dependencies and reviewing
compiler output should pass `-v` to `pip install`.

Openbtmixing package installations can be minimally tested with
```
$ python
>>> import openbtmixing
>>> openbtmixing.__version__
'<version>'
>>> openbtmixing.test()
```

The package can also be built and installed from a clone of this repository with
```
$ cd /path/to/OpenBT/openbtmixing_pypkg
$ python -m build --sdist
$ python -m pip install -v dist/openbtmixing-<version>.tar.gz
```
where we assume that the [build](https://build.pypa.io/en/stable/index.html)
package has already been installed.

Developers can setup a virtual environment with a developer/editable mode
installation of the package with
```
$ /path/to/target/python -m venv /path/to/venv/my_venv
$ . /path/to/venv/my_venv/bin/activate
$ cd /path/to/OpenBT/openbtmixing_pypkg
$ python -m pip install -v -e .
```
In this latter case, the command line tools are built automatically and
installed at `/path/to/OpenBT/openbtmixing_pypkg/src/openbtmixing/bin`.  The
Python package is hardcoded to use those tools so that the existence of another
set of tools in the system and in `PATH` should not cause issues.

## C++ library & command line tool interface
Developers and C++ users can directly build and install the command line tools,
an OpenBTMixing library, and library tests with `tools/build_openbt_clt.sh`.
Note that these do **not** need to be built in order to use the Python package.

## R package
**TODO**: Needs to be written based on current state of affairs.

# Examples

The examples from the article "Model Mixing Using Bayesian Additive Regression Tress" are reproduced in the jupyter noteboook BART_BMM_Technometrics.ipynb. This notebook can be run locally or in a virtual environment such as google colab.
