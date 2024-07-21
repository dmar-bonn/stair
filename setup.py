try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    "tools.libmcubes.mcubes",
    sources=[
        "tools/libmcubes/mcubes.pyx",
        "tools/libmcubes/pywrapper.cpp",
        "tools/libmcubes/marchingcubes.cpp",
    ],
    language="c++",
    extra_compile_args=["-std=c++11"],
    include_dirs=[numpy_include_dir],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    "tools.libmise.mise",
    sources=["tools/libmise/mise.pyx"],
)


# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy_include_dir],
    cmdclass={"build_ext": BuildExtension},
)
