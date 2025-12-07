"""
Optional dependency compatibility layer.

This module handles optional imports for numpy and other dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

# Type checking imports (no runtime cost)
if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    ArrayLike = Union[npt.NDArray[Any], list, tuple]
else:
    ArrayLike = Any

# Runtime numpy detection
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


def to_string_list(items: Any) -> list[str]:
    """
    Convert various input types to a list of strings.

    Handles:
    - Lists and tuples of strings
    - NumPy arrays (when available)
    - Any iterable of string-convertible items
    """
    if HAS_NUMPY and isinstance(items, np.ndarray):
        # Convert numpy array to list of strings
        return [str(x) for x in items.flat]

    if isinstance(items, (list, tuple)):
        return [str(x) for x in items]

    # Generic iterable
    return [str(x) for x in items]


def to_bool_array(results: list[bool]) -> Any:
    """
    Convert a list of bools to numpy array if available, otherwise return list.
    """
    if HAS_NUMPY:
        return np.array(results, dtype=bool)
    return results


def to_int_array(results: list[int]) -> Any:
    """
    Convert a list of ints to numpy array if available, otherwise return list.
    """
    if HAS_NUMPY:
        return np.array(results, dtype=np.int64)
    return results


def to_float_array(results: list[float]) -> Any:
    """
    Convert a list of floats to numpy array if available, otherwise return list.
    """
    if HAS_NUMPY:
        return np.array(results, dtype=np.float64)
    return results
