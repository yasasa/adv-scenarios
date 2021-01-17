import pytest


def pytest_addoption(parser):
    parser.addini('image_comparision_thresh', 'Threshold for MSE image comparision')
    
def pytest_configure(config):
    pytest.image_comparision_thresh = float(config.getini("image_comparision_thresh"))