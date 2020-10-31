from setuptools import setup
from Cython.Build import cythonize
import numpy
import scipy

setup(
    ext_modules = cythonize("iterate_cy.pyx"),
    include_dirs = [numpy.get_include(), scipy.get_include()],
)