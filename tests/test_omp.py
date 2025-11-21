import nx_cpp._nx_cpp as cpp
import os

os.environ["OMP_NUM_THREADS"] = "8"

print("OpenMP threads =", cpp.omp_test())