import collections
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd

from hamilton.base import ResultMixin


class NumpyMatrixResult(ResultMixin):
    """Mixin for building a Numpy Matrix from the result of walking the graph.

    All inputs to the build_result function are expected to be numpy arrays.

    .. code-block:: python

        from hamilton import base, driver

        adapter = base.SimplePythonGraphAdapter(base.NumpyMatrixResult())
        dr = driver.Driver(config, *modules, adapter=adapter)
        numpy_matrix = dr.execute([...], inputs=...)
    """

    @staticmethod
    def build_result(**outputs: Dict[str, Any]) -> np.matrix:
        """Builds a numpy matrix from the passed in, inputs.

        Note: this does not check that the inputs are all numpy arrays/array like things.

        :param outputs: function_name -> np.array.
        :return: numpy matrix
        """
        # TODO check inputs are all numpy arrays/array like things -- else error
        num_rows = -1
        columns_with_lengths = collections.OrderedDict()
        for col, val in outputs.items():  # assumption is fixed order
            if isinstance(val, (int, float)):  # TODO add more things here
                columns_with_lengths[(col, 1)] = val
            else:
                length = len(val)
                if num_rows == -1:
                    num_rows = length
                elif length == num_rows:
                    # we're good
                    pass
                else:
                    raise ValueError(
                        f"Error, got non scalar result that mismatches length of other vector. "
                        f"Got {length} for {col} instead of {num_rows}."
                    )
                columns_with_lengths[(col, num_rows)] = val
        list_of_columns = []
        for (col, length), val in columns_with_lengths.items():
            if length != num_rows and length == 1:
                list_of_columns.append([val] * num_rows)  # expand single values into a full row
            elif length == num_rows:
                list_of_columns.append(list(val))
            else:
                raise ValueError(
                    f"Do not know how to make this column {col} with length {length} have {num_rows} rows"
                )
        # Create the matrix with columns as rows and then transpose
        return np.asmatrix(list_of_columns).T

    def input_types(self) -> List[Type[Type]]:
        """Currently returns anything as numpy types are relatively new and"""
        return [Any]  # Typing

    def output_type(self) -> Type:
        return pd.DataFrame
