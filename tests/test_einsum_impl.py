r"""Tests for einsum."""

import numpy as np
import pytest
from einsum_prototype.einsum_impl import _Dims, Einsum


np.random.seed(31415)
np.set_printoptions(precision=3, suppress=False, threshold=int(1e5), linewidth=200)


def _compute_ref(a_str, b_str, o_str, a, b):
    einsum_str = _Dims(a_str, b_str, o_str).reduce_einsum_str()
    return np.einsum(einsum_str, a, b)


@pytest.mark.parametrize(
    [
        "a_str",
        "b_str",
        "o_str",
        "a_shape",
        "b_shape",
    ],
    [
        pytest.param("m k", "n k", "m n", (2, 3), (4, 3), id="dev"),
    ],
)
def test_einsum(a_str, b_str, o_str, a_shape, b_shape):
    print("-" * 120)
    init_range = 1.0
    a = np.random.uniform(low=-init_range, high=init_range, size=a_shape)
    b = np.random.uniform(low=-init_range, high=init_range, size=a_shape)
    ref = _compute_ref(a_str, b_str, o_str, a, b)
    print(ref)
