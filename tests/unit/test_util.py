from typing import Any

import pytest

from pt2onnx_checker.util import is_iterable_data_structure


@pytest.mark.parametrize(
    "input_case, expected",
    [
        ([1, 2, 3], True),
        ("hello", False),
        ({"a": 1, "b": 2}, True),
        ((1, 2), True),
        (1, False),
        (1.0, False),
    ],
)
def test_is_iterable_and_not_dict(input_case: Any, expected: bool):
    assert is_iterable_data_structure(input_case) == expected
