from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
import os

extensions=[Extension("cystats",["econohmm/cystats.pyx"])]

setup(
    name='econohmm',
    version='',
    packages=[''],
    url='',
    license='',
    author='keithblackwell1',
    author_email='keith.blackwell',
    description='',
    ext_modules=cythonize(os.path.join('econohmm','**','*.pyx')),
    include_dirs=[np.get_include()]
)
