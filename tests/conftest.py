import pytest
import networkx as nx
import os
import random
import time

from road_loader import load_dimacs_graph

# load graph once per test session; reused across all tests

@pytest.fixture(scope="session")
def rome_graph() -> nx.DiGraph:
    path = os.path.join(os.path.dirname(__file__), "rome99.gr")
    if not os.path.isfile(path):
        pytest.fail(f"Rome graph file not found: {path}")
    t0 = time.perf_counter()
    G = load_dimacs_graph(str(path), directed=True)
    t1 = time.perf_counter() - t0
    print("")
    print(f"Rome graph load time = {t1:.3f}s")
    return G

@pytest.fixture(scope="session")
def nyc_graph() -> nx.DiGraph:
    path = os.path.join(os.path.dirname(__file__), "USA-road-d.NY.gr")
    if not os.path.isfile(path):
        pytest.fail(f"NYC graph file not found: {path}")
    t0 = time.perf_counter()
    G = load_dimacs_graph(str(path), directed=True)
    t1 = time.perf_counter() - t0
    print("")
    print(f"NYC graph load time = {t1:.3f}s")
    return G

@pytest.fixture(scope="session")
def usa_ne_graph() -> nx.DiGraph:
    path = os.path.join(os.path.dirname(__file__), "USA-road-d.NE.gr")
    if not os.path.isfile(path):
        pytest.fail(f"USA NE graph file not found: {path}")
    t0 = time.perf_counter()
    G = load_dimacs_graph(str(path), directed=True)
    t1 = time.perf_counter() - t0
    print("")
    print(f"USA NE graph load time = {t1:.3f}s")
    return G

@pytest.fixture(scope="session")
def usa_e_graph() -> nx.DiGraph:
    path = os.path.join(os.path.dirname(__file__), "USA-road-d.E.gr")
    if not os.path.isfile(path):
        pytest.fail(f"USA East graph file not found: {path}")
    t0 = time.perf_counter()
    G = load_dimacs_graph(str(path), directed=True)
    t1 = time.perf_counter() - t0
    print("")
    print(f"USA East graph load time = {t1:.3f}s")
    return G

@pytest.fixture(scope="session")
def rng_seed() -> int:
    """
    session-level random seed
    - if TEST_SEED env var is set, use that to reproduce flaky runs
    - else, generate a random seed each pytest run
    - print the seed so runs can be reproduced
    """
    env_seed = os.getenv("TEST_SEED")
    if env_seed is not None:
        seed = int(env_seed)
        print("")
        print(f"Using TEST_SEED from environment: {seed}")
    else:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        print("")
        print(f"Random seed for this test run: {seed}")

    return seed