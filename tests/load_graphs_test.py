import pytest

# load graphs, runs first alphabetically among test files

@pytest.mark.rome
def test_load_rome(rome_graph):
    _ = rome_graph

@pytest.mark.nyc
def test_load_nyc(nyc_graph):
    _ = nyc_graph

@pytest.mark.usa_ne
def test_load_usa_ne(usa_ne_graph):
    _ = usa_ne_graph

@pytest.mark.usa_e
def test_load_usa_e(usa_e_graph):
    _ = usa_e_graph