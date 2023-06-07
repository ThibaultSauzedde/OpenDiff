import sys
import glob
import os

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11.setup_helpers import ParallelCompile, naive_recompile
from setuptools import setup
from distutils.sysconfig import customize_compiler

__version__ = "0.0.1"


DEBUG = bool(int(os.environ.get('DEBUG', 0)))

if DEBUG:
    optimisation = "-O0"
    os.environ['CFLAGS'] = os.environ['CFLAGS'] + " -g"  #because of pybinf way of addinf -g0
    os.environ['CPPFLAGS'] = os.environ['CPPFLAGS'] + " -g" 
else:
    optimisation = "-O3"
    debug = "-g0"

OPENMP = bool(int(os.environ.get('OPENMP', 1)))

class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        if "-Wstrict-prototypes" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        if "-Wno-unused-result" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-Wno-unused-result")
        if OPENMP:
            self.compiler.compiler_so.append("-fopenmp")

        # replace in compiler
        for step in [self.compiler.compiler_so, self.compiler.linker_so, self.compiler.compiler_cxx]:
            for i in range(len(step)):
                # Intel Pentium by default, change to native, try ‘x86-64’
                if step[i] == "-march=nocona":
                    step[i] = "-march=native"

                # change to native
                if step[i] == "-mtune=haswell":
                    step[i] = "-mtune=native"

                # Optimisation
                if "-O2" in step[i]:
                    # O0 O1 O2 O3 Ofast
                    step[i] = step[i].replace("-O2", optimisation)


        build_ext.build_extensions(self)

# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)


ParallelCompile("NPY_NUM_BUILD_JOBS",
                needs_recompile=naive_recompile).install()


# todo: add petsc as optional, later??
#    + fmt as mandatories libraries 
ext_modules = [
    Pybind11Extension("opendiff",
                      ['opendiff/opendiff.cpp', 'opendiff/materials.cpp',
                       'opendiff/macrolib.cpp',],
                      define_macros=[('VERSION_INFO', __version__)],
                      include_dirs=[
                          os.environ.get('EIGEN_DIR', '/home/ts249161/dev/these/eigen'),
                          os.environ.get('HIGHFIVE_DIR', '/home/ts249161/dev/these/HighFive'),
                          os.environ.get('SPDLOG_DIR', '/home/ts249161/dev/these/spdlog'),
                          os.environ.get('FMT_DIR', '/home/ts249161/dev/these/fmt')],
                      libraries=['petsc', 'slepc', 'fmt'],
                      cxx_std=17),
]

setup(
    name="opendiff",
    version=__version__,
    author="Thibault SAUZEDDE",
    author_email="thibault.sauzedde@pm.me",
    description="A simple solver for the neutron diffusion equation",
    # packages=['opendiff'],
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
    cmdclass={"build_ext": my_build_ext},
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pybind11',
        'pytest',
        'pytest-datadir'],
)

# for a complete example
# https://github.com/spotify/pedalboard
