# ----- SPECIFY PROJECT
# eigen dependency requires at least C++11
#
# The version provided here is the version of the C++ library and command line
# tools.  The Python package maintains a separate version.  If the C++ code is
# altered, then the version number here should be manually altered and the
# Python package's version number altered in a similar way since changes made to
# the C++ code can affect the behavior of the Python code.  If, however, only
# the Python code is changed, only the version of the Python package should be
# altered.
project('openbtmixing', 'cpp',
        meson_version: '>=1.6.0',
        version: '1.0.1',
        license: 'MIT',
        default_options: [
          'cpp_std=c++11',
          'buildtype=debug',
          'default_library=shared',
          'b_ndebug=if-release',
          'b_pch=false',
          'unity=off',
          'verbose=true',
          'use_mpi=true',
          'pypkg=false'
        ])

# ----- DEPENDENCIES
# If users have a local install of eigen that can be discovered with pkg-config
# or other Meson tricks, then that installation will be used.  Otherwise, meson
# will download the specific tarball as defined in eigen.wrap and use that.
eigen_dep = dependency('eigen3', fallback: ['eigen', 'eigen_dep'])
deps_all = [eigen_dep]

# ----- USER ARGUMENTS
# Run this after dependencies so that we can alter dependencies according to
# the user's needs.
if not get_option('verbose')
    add_project_arguments('-DSILENT', language: 'cpp')
endif

if get_option('use_mpi')
    add_project_arguments('-D_OPENMPI', language: 'cpp')
    # This works with both OpenMPI & MPICH beginning with v1.6.0.  See
    # https://github.com/mesonbuild/meson/pull/13619
    deps_all += [dependency('mpi', language: 'cpp')]
endif

# ----- SPECIFY HEADERS USED TO BUILD LIBRARY, CLTs, & TESTS
incdir = include_directories('includes', is_system: false)

# ----- SPECIFY libopenbtmixing LIBRARY
# Sources to build individual components in single library
src_crn      = ['src/crn.cpp', 'src/tnorm.cpp']
src_tree     = ['src/treefuns.cpp', 'src/tree.cpp']
src_brt      = ['src/brt.cpp', 'src/brtmoves.cpp', 'src/brtfuns.cpp']
src_mbrt     = ['src/mbrt.cpp']
src_sbrt     = ['src/sbrt.cpp']
src_mxbrt    = ['src/mxbrt.cpp']
src_ambrt    = ['src/ambrt.cpp']
src_psbrt    = ['src/psbrt.cpp']
src_amxbrt   = ['src/amxbrt.cpp']
src_poisson  = ['src/singlepoisson.cpp']
src_binomial = ['src/singlebinomial.cpp']

lib_srcs  = src_crn + src_tree + src_brt + src_mbrt + src_sbrt
lib_srcs += src_mxbrt + src_ambrt + src_psbrt + src_amxbrt
lib_srcs += src_poisson + src_binomial

# ----- SPECIFY LIBRARY UNIT TESTS
# Only include those tests that presently build & run through successfully.
tests_all  = ['crn', 'tree']
#tests_all += ['brt', 'brtvp']
#tests_all += ['mbrt', 'ambrt', 'mxbrt', 'amxbrt']
#tests_all += ['sbrt', 'psbrt']
#tests_all += ['singlepoisson', 'singlebinomial']

# ----- SPECIFY COMMAND LINE TOOLS
CLTs = [['openbtcli',        'src/cli.cpp'],
        ['openbtpred',       'src/pred.cpp'],
        ['openbtvartivity',  'src/vartivity.cpp'],
        ['openbtsobol',      'src/sobol.cpp'],
        ['openbtmopareto',   'src/mopareto.cpp'],
        ['openbtmixingwts',  'src/mixingwts.cpp'],
        ['openbtmixing',     'src/mixandemulate.cpp'],
        ['openbtmixingpred', 'src/mixandemulatepred.cpp']]

# ----- BUILD & INSTALL ALL PRODUCTS
# For the Python package, only the command line tools are needed since these are
# built as standalone binaries rather than linked against the openbtmixing
# library.  This will keep the package's source distribution and installation as
# small and clean as possible.
if not get_option('pypkg')
    # Install the headers along with library for building against library
    install_headers('includes/ambrt.h',
                    'includes/amxbrt.h',
                    'includes/brt.h',
                    'includes/brtfuns.h',
                    'includes/crn.h',
                    'includes/dinfo.h',
                    'includes/mbrt.h',
                    'includes/mxbrt.h',
                    'includes/psbrt.h',
                    'includes/rn.h',
                    'includes/sbrt.h',
                    'includes/singlebinomial.h',
                    'includes/singlepoisson.h',
                    'includes/tnorm.h',
                    'includes/tree.h',
                    'includes/treefuns.h')

    lib = library('openbtmixing', sources: lib_srcs,
                  dependencies: deps_all,
                  include_directories: incdir,
                  version: meson.project_version(),
                  darwin_versions: meson.project_version(),
                  pic: true,
                  install: true)

    # Build up test suite for testing with `meson test` in build directory
    foreach name : tests_all
        test_name = f'test_@name@' 
        test_src = f'tests/test_@name@.cpp'
        test_exe = executable(test_name, test_src,
                              dependencies: deps_all, link_with: lib,
                              include_directories: incdir,
                              install: false)
        test(name, test_exe)
    endforeach
endif

# All build variants get standalone CLT binaries
foreach each : CLTs
    executable(each[0], [each[1]] + lib_srcs,
               dependencies: deps_all,
               include_directories: incdir,
               install: true)
endforeach
