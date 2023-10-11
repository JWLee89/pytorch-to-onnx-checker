from typing import Any, Iterable


def is_iterable_data_structure(obj: Any) -> bool:
    """Given an input, check whether an object is iterable or not. We will omit strings, since
    strings are different in comparison to other iterables such as tuple or list, in terms of data
    structure type.

    Args:
        obj (Any): The object to iterate over

    Returns:
        bool: True if object is iterable and not a string.
    """
    return isinstance(obj, Iterable) and not isinstance(obj, str)
