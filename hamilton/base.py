"""This module contains base constructs for executing a hamilton graph.
It should only import hamilton.node, numpy, pandas.
It cannot import hamilton.graph, or hamilton.driver.
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Type

from hamilton.lifecycle import api as lifecycle_api

logger = logging.getLogger(__name__)


class ResultMixin(lifecycle_api.LegacyResultMixin):
    """Legacy result builder -- see lifecycle methods for more information."""

    pass


class DictResult(ResultMixin):
    """Simple function that returns the dict of column -> value results.

    It returns the results as a dictionary, where the keys map to outputs requested,
    and values map to what was computed for those values.

    Use this when you want to:

       1. debug dataflows.
       2. have heterogeneous return types.
       3. Want to manually transform the result into something of your choosing.


    .. code-block:: python

        from hamilton import base, driver

        dict_builder = base.DictResult()
        adapter = base.SimplePythonGraphAdapter(dict_builder)
        dr = driver.Driver(config, *modules, adapter=adapter)
        dict_result = dr.execute([...], inputs=...)

    Note, if you just want the dict result + the SimplePythonGraphAdapter, you can use the
    DefaultAdapter

    .. code-block:: python

        adapter = base.DefaultAdapter()
    """

    @staticmethod
    def build_result(**outputs: Dict[str, Any]) -> Dict:
        """This function builds a simple dict of output -> computed values."""
        return outputs

    def input_types(self) -> Optional[List[Type[Type]]]:
        return [Any]

    def output_type(self) -> Type:
        return Dict[str, Any]


class HamiltonGraphAdapter(lifecycle_api.GraphAdapter, abc.ABC):
    """Legacy graph adapter -- see lifecycle methods for more information."""

    pass
