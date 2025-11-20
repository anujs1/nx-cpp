import pytest

# load graphs, runs first alphabetically among test files

@pytest.mark.nyc
def test_load_nyc(nyc_graph):
    _ = nyc_graph

# @pytest.mark.usa
# def test_load_usa(usa_graph):
#     _ = usa_graph