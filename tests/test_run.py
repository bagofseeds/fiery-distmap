import pytest
from utils import make_ndsphere

from fiery.distmap.l1 import l1_signed_transform
from fiery.distmap.l2 import euclidean_signed_transform


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_run_l1(ndim):
    x = make_ndsphere(ndim)
    y = l1_signed_transform(x)
    assert y.shape == x.shape
    assert y.dtype.is_floating_point


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_run_l2(ndim):
    x = make_ndsphere(ndim)
    y = euclidean_signed_transform(x)
    assert y.shape == x.shape
    assert y.dtype.is_floating_point
