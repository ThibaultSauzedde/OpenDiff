import sys
import glob

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from distutils.sysconfig import customize_compiler

__version__ = "0.0.1"



class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("opendiff",
                      ['opendiff/opendiff.cpp', 'opendiff/materials.cpp',
                       'opendiff/macrolib.cpp'],
                      define_macros=[('VERSION_INFO', __version__)],
                      include_dirs=[
                          '/home/ts249161/anaconda3/envs/opendiff/include/eigen3'],
                    #   library_dirs=[
                    #       "/home/ts249161/anaconda3/envs/opendiff/x86_64-conda-linux-gnu/sysroot/lib/"],
                    #   extra_link_args=[
                    #       "-Wl,-rpath=/home/ts249161/anaconda3/envs/opendiff/x86_64-conda-linux-gnu/sysroot/lib/"],
                      cxx_std=17),
]

setup(
    name="opendiff",
    version=__version__,
    author="Thibault SAUZEDDE",
    author_email="thibault.sauzedde@proton.me",
    description="A simple solver for the neutron diffusion equation",
    packages=['opendiff'],
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
    cmdclass={"build_ext": my_build_ext},
    python_requires=">=3.6",
)


#https://github.com/spotify/pedalboard
