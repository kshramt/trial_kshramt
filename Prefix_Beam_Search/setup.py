from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("prefix_beam_search.py"),
    include_dirs=[np.get_include()],
)
