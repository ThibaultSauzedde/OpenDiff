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

class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        self.compiler.compiler_so.remove("-Wstrict-prototypes")
        self.compiler.compiler_so.remove("-Wno-unused-result")

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
# spdlog + fmt as mandatories libraries 
ext_modules = [
    Pybind11Extension("opendiff",
                      ['opendiff/opendiff.cpp', 'opendiff/materials.cpp',
                       'opendiff/macrolib.cpp',],
                      define_macros=[('VERSION_INFO', __version__)],
                      include_dirs=[
                          os.environ.get(
                              'EIGEN_DIR', '/home/ts249161/dev/these/eigen'),
                          os.environ.get('HIGHFIVE_DIR', '/home/ts249161/dev/these/HighFive')],
                      libraries=['petsc', 'slepc'],
                      # ajouter macro pour slepc ou spectra -DNOM_MACRO
                      #   extra_compile_args=["-O3"],
                      #   library_dirs=[
                      #       "/home/ts249161/anaconda3/envs/opendiff/x86_64-conda-linux-gnu/sysroot/lib/"],
                      #   extra_link_args=[
                      #       "-Wl,-rpath=/home/ts249161/anaconda3/envs/opendiff/x86_64-conda-linux-gnu/sysroot/lib/"],
                      # extra_link_args=[
                      #     "-Wl,--no-undefined"],
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
)

# for a complete example
# https://github.com/spotify/pedalboard
