import os
import sys
import shutil
import codecs

import subprocess as sbp

from pathlib import Path
from setuptools import (
    setup, Command
)
from setuptools.command.build import build as _build

# ----- HARDCODED VALUES
PKG_ROOT = Path(__file__).resolve().parent
PY_SRC_PATH = PKG_ROOT.joinpath("src", "openbtmixing")
CLT_SRC_PATH = PKG_ROOT.joinpath("cpp")

# Names of C++ products to include
#
# Only a subset of OpenBT command line tools are used in package.  Any changes
# made here should also be made in interface_helpers.py:run_model.
CLT_NAMES = ["openbtcli", "openbtpred", "openbtmixingwts"]

# Package metadata
PYTHON_REQUIRES = ">=3.9"
CODE_REQUIRES = ["setuptools", "numpy", "matplotlib"]
TEST_REQUIRES = ["pytest", "scipy", "pandas"]
INSTALL_REQUIRES = CODE_REQUIRES + TEST_REQUIRES

PACKAGE_DATA = {
    "openbtmixing":
        ["tests/bart_bmm_test_data/2d_*.txt"] +
        [f"bin/{clt}" for clt in CLT_NAMES]
}

PROJECT_URLS = {
    "Source": "https://github.com/jcyannotty/OpenBT",
    "Documentation": "https://github.com/jcyannotty/OpenBT",
    "Tracker": "https://github.com/jcyannotty/OpenBT/issues",
}


# ----- CUSTOM COMMAND TO BUILD C++ CLTs
# EdgeDB builds a CLT that they make available through their interface.  This
# is partially based on what they have done at commit c5db98c.
#
# https://github.com/edgedb/edgedb/blob/master/setup.py
class build(_build):
    sub_commands = ([("build_clt", None)])


class build_clt(Command):
    description = "Build the OpenBTMixing command line tools"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self, *args, **kwargs):
        if shutil.which("meson", mode=os.F_OK | os.X_OK) is None:
            print()
            print("Please install the Meson build system & add meson to path")
            print()
            sys.exit(1)

        # To build debug versions with more output,
        # * use --buildtype=debug
        # * consider using -Dverbose=true
        # * consider adding arguments such as --warnlevel and --werror to
        #   SETUP_CMD
        # * Remove "--quiet" from INSTALL_CMD
        # * Use python -m pip install -v ...
        SETUP_CMD = ["meson", "setup", "--wipe", "--clearcache",
                     "--buildtype=release", "builddir", f"-Dprefix={PY_SRC_PATH}",
                     "-Duse_mpi=true", "-Dverbose=false", "-Dpypkg=true"]
        COMPILE_CMD = ["meson", "compile", "-v", "-C", "builddir"]
        INSTALL_CMD = ["meson", "install", "--quiet", "-C", "builddir"]

        # Install the CLTs within the Python source files and so that they are
        # included in the wheel build based on PACKAGE_DATA
        cwd = Path.cwd()
        os.chdir(CLT_SRC_PATH)
        for cmd in [SETUP_CMD, COMPILE_CMD, INSTALL_CMD]:
            try:
                sbp.run(cmd, stdin=sbp.DEVNULL, capture_output=False, check=True)
            except sbp.CalledProcessError as err:
                print()
                msg = "[meson build] Unable to run command (Return code {})"
                print(msg.format(err.returncode))
                print("[meson build] " + " ".join(err.cmd))
                sys.exit(2)
        os.chdir(cwd)


cmdclass = {
    'build': build,
    'build_clt': build_clt
}


# ----- SPECIFY THE PACKAGE
def readme_md():
    fname = PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, "r", encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


setup(
    name='openbtmixing',
    version=version(),
    author="John Yannotty",
    author_email="yannotty.1@buckeyemail.osu.edu",
    maintainer="John Yannotty",
    maintainer_email="yannotty.1@buckeyemail.osu.edu",
    license="MIT",
    package_dir={"": "src"},
    package_data=PACKAGE_DATA,
    cmdclass=cmdclass,
    url=PROJECT_URLS["Source"],
    project_urls=PROJECT_URLS,
    description="Model mixing using Bayesian Additive Regression Trees",
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
