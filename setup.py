from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

extensions=[Extension("econohmm",["cystats.pyx"])]

setup(
    name='econohmm',
    version='',
    packages=[''],
    url='',
    license='',
    author='keithblackwell1',
    author_email='keith.blackwell',
    description=''
)
