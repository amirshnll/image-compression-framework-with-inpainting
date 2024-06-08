import json
import numpy as np


class TypeCaster:
    """
    The TypeCaster class provides methods to convert various data types to strings and vice versa.
    It includes functions to handle tuples, integers, NumPy arrays, and dictionaries. Additionally,
    it provides a utility to convert 1D NumPy arrays into 2D arrays with a specific transformation.
    """

    def tuple_to_str(self, t: tuple) -> str:
        """
        Convert a tuple to a string.
        """
        return str(t)

    def str_to_tuple(self, s: str) -> tuple:
        """
        Convert a string to a tuple.
        """
        return eval(s)

    def int_to_str(self, i: int) -> str:
        """
        Convert an integer to a string.
        """
        return str(i)

    def str_to_int(self, s: str) -> int:
        """
        Convert a string to an integer.
        """
        return int(s)

    def ndarray_to_str(self, arr: np.ndarray) -> str:
        """
        Convert a NumPy array to a string.
        """
        return json.dumps(
            {"data": arr.tolist(), "dtype": str(arr.dtype), "shape": arr.shape}
        )

    def str_to_ndarray(self, s: str) -> np.ndarray:
        """
        Convert a string to a NumPy array.
        """
        obj = json.loads(s)
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    def dict_to_str(self, d: dict) -> str:
        """
        Convert a dictionary to a JSON-formatted string.
        """
        return json.dumps(d)

    def str_to_dict(self, s: str) -> dict:
        """
        Convert a JSON-formatted string to a dictionary.
        """
        return json.loads(s)
