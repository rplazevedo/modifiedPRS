from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["sprs_init.pyx", "sprs_cont.pyx", "merge_v.pyx", "merge_phi.pyx"],annotate=True),
    include_dirs=[numpy.get_include()]
)


#python3 setup.py build_ext --inplace
