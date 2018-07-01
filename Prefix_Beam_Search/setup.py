from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

print(np.get_include())

setup(
    ext_modules=cythonize("prefix_beam_search.py"),
    include_dirs=[np.get_include()],
)
