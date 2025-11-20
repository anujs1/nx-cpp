from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

compile_args = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
link_args = []
system = platform.system()
if system == "Linux":
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]
elif system == "Darwin":
    # Requires 'brew install libomp' present on the system.
    compile_args += ["-Xpreprocessor", "-fopenmp"]
    link_args += ["-lomp"]

ext_modules = [
    Pybind11Extension(
        "nx_cpp._nx_cpp",
        ["nx_cpp/_core.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="nx-cpp",
    version="0.0.1",
    description="Minimal C++ backend for NetworkX (demo)",
    packages=["nx_cpp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.2",
        "pybind11>=2.11",
        "numpy>=1.21",
        "scipy>=1.9",
    ],
    entry_points={
        "networkx.backends": ["cpp = nx_cpp.backend:backend"],
        "networkx.backend_info": ["cpp = nx_cpp.backend:get_info"],
    },
)
