import collections
import logging
import sys
from types import ModuleType
from typing import Any, Callable, Collection, Dict, List, Tuple, Type, Union, get_type_hints

import pandas as pd
from pandas.core.indexes import extension as pd_extension

from hamilton.base import DictResult, HamiltonGraphAdapter, ResultMixin, logger

_sys_version_info = sys.version_info
_version_tuple = (_sys_version_info.major, _sys_version_info.minor, _sys_version_info.micro)

if _version_tuple < (3, 11, 0):
    pass
else:
    pass

from hamilton import htypes, node, registry
from hamilton.function_modifiers.expanders import extract_columns
from hamilton.function_modifiers.recursive import (
    _default_inject_parameter,
    subdag,
    with_columns_base,
)
from hamilton.plugins.pandas_extensions import DATAFRAME_TYPE


class with_columns(with_columns_base):
    """Initializes a with_columns decorator for pandas. This allows you to efficiently run groups of map operations on a dataframe.

    Here's an example of calling it -- if you've seen ``@subdag``, you should be familiar with
    the concepts:

    .. code-block:: python

        # my_module.py
        def a(a_from_df: pd.Series) -> pd.Series:
            return _process(a)

        def b(b_from_df: pd.Series) -> pd.Series:
            return _process(b)

        def a_b_average(a_from_df: pd.Series, b_from_df: pd.Series) -> pd.Series:
            return (a_from_df + b_from_df) / 2


    .. code-block:: python

        # with_columns_module.py
        def a_plus_b(a: pd.Series, b: pd.Series) -> pd.Series:
            return a + b


        # the with_columns call
        @with_columns(
            *[my_module], # Load from any module
            *[a_plus_b], # or list operations directly
            columns_to_pass=["a_from_df", "b_from_df"], # The columns to pass from the dataframe to
            # the subdag
            select=["a", "b", "a_plus_b", "a_b_average"], # The columns to select from the dataframe
        )
        def final_df(initial_df: pd.DataFrame) -> pd.DataFrame:
            # process, or just return unprocessed
            ...

    In this instance the ``initial_df`` would get two columns added: ``a_plus_b`` and ``a_b_average``.

    The operations are applied in topological order. This allows you to
    express the operations individually, making it easy to unit-test and reuse.

    Note that the operation is "append", meaning that the columns that are selected are appended
    onto the dataframe.

    If the function takes multiple dataframes, the dataframe input to process will always be
    the first argument. This will be passed to the subdag, transformed, and passed back to the function.
    This follows the hamilton rule of reference by parameter name. To demonstarte this, in the code
    above, the dataframe that is passed to the subdag is `initial_df`. That is transformed
    by the subdag, and then returned as the final dataframe.

    You can read it as:

    "final_df is a function that transforms the upstream dataframe initial_df, running the transformations
    from my_module. It starts with the columns a_from_df and b_from_df, and then adds the columns
    a, b, and a_plus_b to the dataframe. It then returns the dataframe, and does some processing on it."

    In case you need more flexibility you can alternatively use ``on_input``, for example,

    .. code-block:: python

        # with_columns_module.py
        def a_from_df(initial_df: pd.Series) -> pd.Series:
            return initial_df["a_from_df"] / 100

        def b_from_df(initial_df: pd.Series) -> pd.Series:
            return initial_df["b_from_df"] / 100


        # the with_columns call
        @with_columns(
            *[my_module],
            *[a_from_df],
            on_input="initial_df",
            select=["a_from_df", "b_from_df", "a", "b", "a_plus_b", "a_b_average"],
        )
        def final_df(initial_df: pd.DataFrame, ...) -> pd.DataFrame:
            # process, or just return unprocessed
            ...

    the above would output a dataframe where the two columns ``a_from_df`` and ``b_from_df`` get
    overwritten.
    """

    def __init__(
        self,
        *load_from: Union[Callable, ModuleType],
        columns_to_pass: List[str] = None,
        pass_dataframe_as: str = None,
        on_input: str = None,
        select: List[str] = None,
        namespace: str = None,
        config_required: List[str] = None,
    ):
        """Instantiates a ``@with_columns`` decorator.

        :param load_from: The functions or modules that will be used to generate the group of map operations.
        :param columns_to_pass: The initial schema of the dataframe. This is used to determine which
            upstream inputs should be taken from the dataframe, and which shouldn't. Note that, if this is
            left empty (and external_inputs is as well), we will assume that all dependencies come
            from the dataframe. This cannot be used in conjunction with on_input.
        :param on_input: The name of the dataframe that we're modifying, as known to the subdag.
            If you pass this in, you are responsible for extracting columns out. If not provided, you have
            to pass columns_to_pass in, and we will extract the columns out on the first parameter for you.
        :param select: The end nodes that represent columns to be appended to the original dataframe
            via with_columns. Existing columns will be overridden. The selected nodes need to have the
            corresponding column type, in this case pd.Series, to be appended to the original dataframe.
        :param namespace: The namespace of the nodes, so they don't clash with the global namespace
            and so this can be reused. If its left out, there will be no namespace (in which case you'll want
            to be careful about repeating it/reusing the nodes in other parts of the DAG.)
        :param config_required: the list of config keys that are required to resolve any functions. Pass in None\
            if you want the functions/modules to have access to all possible config.
        """

        if pass_dataframe_as is not None:
            raise NotImplementedError(
                "We currently do not support pass_dataframe_as for pandas. Please reach out if you need this "
                "functionality."
            )

        super().__init__(
            *load_from,
            columns_to_pass=columns_to_pass,
            on_input=on_input,
            select=select,
            namespace=namespace,
            config_required=config_required,
            dataframe_type=DATAFRAME_TYPE,
        )

    def _create_column_nodes(
        self, fn: Callable, inject_parameter: str, params: Dict[str, Type[Type]]
    ) -> List[node.Node]:
        output_type = params[inject_parameter]

        def temp_fn(**kwargs) -> Any:
            return kwargs[inject_parameter]

        # We recreate the df node to use extract columns
        temp_node = node.Node(
            name=inject_parameter,
            typ=output_type,
            callabl=temp_fn,
            input_types={inject_parameter: output_type},
        )

        extract_columns_decorator = extract_columns(*self.initial_schema)

        out_nodes = extract_columns_decorator.transform_node(temp_node, config={}, fn=temp_fn)
        return out_nodes[1:]

    def get_initial_nodes(
        self, fn: Callable, params: Dict[str, Type[Type]]
    ) -> Tuple[str, Collection[node.Node]]:
        """Selects the correct dataframe and optionally extracts out columns."""
        inject_parameter = _default_inject_parameter(fn=fn, target_dataframe=self.target_dataframe)
        with_columns_base.validate_dataframe(
            fn=fn,
            inject_parameter=inject_parameter,
            params=params,
            required_type=self.dataframe_type,
        )

        initial_nodes = (
            []
            if self.target_dataframe is not None
            else self._create_column_nodes(fn=fn, inject_parameter=inject_parameter, params=params)
        )

        return inject_parameter, initial_nodes

    def get_subdag_nodes(self, fn: Callable, config: Dict[str, Any]) -> Collection[node.Node]:
        return subdag.collect_nodes(config, self.subdag_functions)

    def chain_subdag_nodes(
        self, fn: Callable, inject_parameter: str, generated_nodes: Collection[node.Node]
    ) -> node.Node:
        "Node that adds to / overrides columns for the original dataframe based on selected output."
        # In case no node is selected we append all possible nodes that have a column type matching
        # what the dataframe expects
        if self.select is None:
            self.select = [
                sink_node.name
                for sink_node in generated_nodes
                if sink_node.type == registry.get_column_type_from_df_type(self.dataframe_type)
            ]

        def new_callable(**kwargs) -> Any:
            df = kwargs[inject_parameter]
            columns_to_append = {}
            for column in self.select:
                columns_to_append[column] = kwargs[column]

            return df.assign(**columns_to_append)

        column_type = registry.get_column_type_from_df_type(self.dataframe_type)
        input_map = {column: column_type for column in self.select}
        input_map[inject_parameter] = self.dataframe_type
        merge_node = node.Node(
            name="_append",
            typ=self.dataframe_type,
            callabl=new_callable,
            input_types=input_map,
        )
        output_nodes = generated_nodes + [merge_node]
        return output_nodes, merge_node.name

    def validate(self, fn: Callable):
        inject_parameter = _default_inject_parameter(fn=fn, target_dataframe=self.target_dataframe)
        params = get_type_hints(fn)
        with_columns_base.validate_dataframe(
            fn=fn,
            inject_parameter=inject_parameter,
            params=params,
            required_type=self.dataframe_type,
        )


class PandasDataFrameResult(ResultMixin):
    """Mixin for building a pandas dataframe from the result.

    It returns the results as a Pandas Dataframe, where the columns map to outputs requested, and values map to what\
    was computed for those values. Note: this only works if the computed values are pandas series, or scalar values.

    Use this when you want to create a pandas dataframe.

    Example:

    .. code-block:: python

        from hamilton import base, driver
        df_builder = base.PandasDataFrameResult()
        adapter = base.SimplePythonGraphAdapter(df_builder)
        dr =  driver.Driver(config, *modules, adapter=adapter)
        df = dr.execute([...], inputs=...)
    """

    @staticmethod
    def pandas_index_types(
        outputs: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """This function creates three dictionaries according to whether there is an index type or not.

        The three dicts we create are:
        1. Dict of index type to list of outputs that match it.
        2. Dict of time series / categorical index types to list of outputs that match it.
        3. Dict of `no-index` key to list of outputs with no index type.

        :param outputs: the dict we're trying to create a result from.
        :return: dict of all index types, dict of time series/categorical index types, dict if there is no index
        """
        all_index_types = collections.defaultdict(list)
        time_indexes = collections.defaultdict(list)
        no_indexes = collections.defaultdict(list)

        def index_key_name(pd_object: Union[pd.DataFrame, pd.Series]) -> str:
            """Creates a string helping identify the index and it's type.
            Useful for disambiguating time related indexes."""
            return f"{pd_object.index.__class__.__name__}:::{pd_object.index.dtype}"

        def get_parent_time_index_type():
            """Helper to pull the right time index parent class."""
            if hasattr(pd_extension, "NDArrayBackedExtensionIndex"):
                index_type = pd_extension.NDArrayBackedExtensionIndex
            else:
                index_type = None  # weird case, but not worth breaking for.
            return index_type

        for output_name, output_value in outputs.items():
            if isinstance(
                output_value, (pd.DataFrame, pd.Series)
            ):  # if it has an index -- let's grab it's type
                dict_key = index_key_name(output_value)
                if isinstance(output_value.index, get_parent_time_index_type()):
                    # it's a time index -- these will produce garbage if not aligned properly.
                    time_indexes[dict_key].append(output_name)
            elif isinstance(
                output_value, pd.Index
            ):  # there is no index on this - so it's just an integer one.
                int_index = pd.Series(
                    [1, 2, 3], index=[0, 1, 2]
                )  # dummy to get right values for string.
                dict_key = index_key_name(int_index)
            else:
                dict_key = "no-index"
                no_indexes[dict_key].append(output_name)
            all_index_types[dict_key].append(output_name)
        return all_index_types, time_indexes, no_indexes

    @staticmethod
    def check_pandas_index_types_match(
        all_index_types: Dict[str, List[str]],
        time_indexes: Dict[str, List[str]],
        no_indexes: Dict[str, List[str]],
    ) -> bool:
        """Checks that pandas index types match.

        This only logs warning errors, and if debug is enabled, a debug statement to list index types.
        """
        no_index_length = len(no_indexes)
        time_indexes_length = len(time_indexes)
        all_indexes_length = len(all_index_types)
        number_with_indexes = all_indexes_length - no_index_length
        types_match = True  # default to True
        # if there is more than one time index
        if time_indexes_length > 1:
            logger.warning(
                "WARNING: Time/Categorical index type mismatches detected - check output to ensure Pandas "
                "is doing what you intend to do. Else change the index types to match. Set logger to debug "
                "to see index types."
            )
            types_match = False
        # if there is more than one index type and it's not explained by the time indexes then
        if number_with_indexes > 1 and all_indexes_length > time_indexes_length:
            logger.warning(
                "WARNING: Multiple index types detected - check output to ensure Pandas is "
                "doing what you intend to do. Else change the index types to match. Set logger to debug to "
                "see index types."
            )
            types_match = False
        elif number_with_indexes == 1 and no_index_length > 0:
            logger.warning(
                f"WARNING: a single pandas index was found, but there are also {len(no_indexes['no-index'])} "
                "outputs without an index. Please check whether the dataframe created matches what what you "
                "expect to happen."
            )
            # Strictly speaking the index types match -- there is only one -- so setting to True.
            types_match = True
        # if all indexes matches no indexes
        elif no_index_length == all_indexes_length:
            logger.warning(
                "It appears no Pandas index type was detected (ignore this warning if you're using DASK for now.) "
                "Please check whether the dataframe created matches what what you expect to happen."
            )
            types_match = False
        if logger.isEnabledFor(logging.DEBUG):
            import pprint

            pretty_string = pprint.pformat(dict(all_index_types))
            logger.debug(f"Index types encountered:\n{pretty_string}.")
        return types_match

    @staticmethod
    def build_result(**outputs: Dict[str, Any]) -> pd.DataFrame:
        """Builds a Pandas DataFrame from the outputs.

        This function will check the index types of the outputs, and log warnings if they don't match.
        The behavior of pd.Dataframe(outputs) is that it will do an outer join based on indexes of the Series passed in.

        :param outputs: the outputs to build a dataframe from.
        """
        # TODO check inputs are pd.Series, arrays, or scalars -- else error
        output_index_type_tuple = PandasDataFrameResult.pandas_index_types(outputs)
        # this next line just log warnings
        # we don't actually care about the result since this is the current default behavior.
        PandasDataFrameResult.check_pandas_index_types_match(*output_index_type_tuple)

        if len(outputs) == 1:
            (value,) = outputs.values()  # this works because it's length 1.
            if isinstance(value, pd.DataFrame):
                return value

        if not any(pd.api.types.is_list_like(value) for value in outputs.values()):
            # If we're dealing with all values that don't have any "index" that could be created
            # (i.e. scalars, objects) coerce the output to a single-row, multi-column dataframe.
            return pd.DataFrame([outputs])
        #
        contains_df = any(isinstance(value, pd.DataFrame) for value in outputs.values())
        if contains_df:
            # build the dataframe from the outputs
            return PandasDataFrameResult.build_dataframe_with_dataframes(outputs)
        # don't do anything special if dataframes aren't in the output.
        return pd.DataFrame(outputs)  # this does an implicit outer join based on index.

    @staticmethod
    def build_dataframe_with_dataframes(outputs: Dict[str, Any]) -> pd.DataFrame:
        """Builds a dataframe from the outputs in an "outer join" manner based on index.

        The behavior of pd.Dataframe(outputs) is that it will do an outer join based on indexes of the Series passed in.
        To handle dataframes, we unpack the dataframe into a dict of series, check to ensure that no columns are
        redefined in a rolling fashion going in order of the outputs requested. This then results in an "enlarged"
        outputs dict that is then passed to pd.Dataframe(outputs) to get the final dataframe.

        :param outputs: The outputs to build the dataframe from.
        :return: A dataframe with the outputs.
        """

        def get_output_name(output_name: str, column_name: str) -> str:
            """Add function prefix to columns.
            Note this means that they stop being valid python identifiers due to the `.` in the string.
            """
            return f"{output_name}.{column_name}"

        flattened_outputs = {}
        for name, output in outputs.items():
            if isinstance(output, pd.DataFrame):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Unpacking dataframe {name} into dict of series with columns {list(output.columns)}."
                    )

                df_dict = {
                    get_output_name(name, col_name): col_value
                    for col_name, col_value in output.to_dict(orient="series").items()
                }
                flattened_outputs.update(df_dict)
            elif isinstance(output, pd.Series):
                if name in flattened_outputs:
                    raise ValueError(
                        f"Series {name} already exists in the output. "
                        f"Please rename the series to avoid this error, or determine from where the initial series is "
                        f"being added; it may be coming from a dataframe that is being unpacked."
                    )
                flattened_outputs[name] = output
            else:
                if name in flattened_outputs:
                    raise ValueError(
                        f"Non series output {name} already exists in the output. "
                        f"Please rename this output to avoid this error, or determine from where the initial value is "
                        f"being added; it may be coming from a dataframe that is being unpacked."
                    )
                flattened_outputs[name] = output

        return pd.DataFrame(flattened_outputs)

    def input_types(self) -> List[Type[Type]]:
        """Currently this just shoves anything into a dataframe. We should probably
        tighten this up."""
        return [Any]

    def output_type(self) -> Type:
        return pd.DataFrame


class StrictIndexTypePandasDataFrameResult(PandasDataFrameResult):
    """A ResultBuilder that produces a dataframe only if the index types match exactly.

    Note: If there is no index type on some outputs, e.g. the value is a scalar, as long as there exists a single \
    pandas index type, no error will be thrown, because a dataframe can be easily created.

    Use this when you want to create a pandas dataframe from the outputs, but you want to ensure that the index types \
    match exactly.

    To use:

    .. code-block:: python

        from hamilton import base, driver
        strict_builder = base.StrictIndexTypePandasDataFrameResult()
        adapter = base.SimplePythonGraphAdapter(strict_builder)
        dr =  driver.Driver(config, *modules, adapter=adapter)
        df = dr.execute([...], inputs=...)  # this will now error if index types mismatch.
    """

    @staticmethod
    def build_result(**outputs: Dict[str, Any]) -> pd.DataFrame:
        # TODO check inputs are pd.Series, arrays, or scalars -- else error
        output_index_type_tuple = PandasDataFrameResult.pandas_index_types(outputs)
        indexes_match = PandasDataFrameResult.check_pandas_index_types_match(
            *output_index_type_tuple
        )
        if not indexes_match:
            import pprint

            pretty_string = pprint.pformat(dict(output_index_type_tuple[0]))
            raise ValueError(
                "Error: pandas index types did not match exactly. "
                f"Found the following indexes:\n{pretty_string}"
            )

        return PandasDataFrameResult.build_result(**outputs)


class SimplePythonDataFrameGraphAdapter(HamiltonGraphAdapter, PandasDataFrameResult):
    """This is the original Hamilton graph adapter. It uses plain python and builds a dataframe result.

    This executes the Hamilton dataflow locally on a machine in a single threaded,
    single process fashion. It assumes a pandas dataframe as a result.

    Use this when you want to execute on a single machine, without parallelization, and you want a
    pandas dataframe as output.
    """

    @staticmethod
    def check_input_type(node_type: Type, input_value: Any) -> bool:
        return htypes.check_input_type(node_type, input_value)

    @staticmethod
    def check_node_type_equivalence(node_type: Type, input_type: Type) -> bool:
        return node_type == input_type

    def execute_node(self, node: node.Node, kwargs: Dict[str, Any]) -> Any:
        return node.callable(**kwargs)


class SimplePythonGraphAdapter(SimplePythonDataFrameGraphAdapter):
    """This class allows you to swap out the build_result very easily.

    This executes the Hamilton dataflow locally on a machine in a single threaded, single process fashion. It allows\
    you to specify a ResultBuilder to control the return type of what ``execute()`` returns.

    Currently this extends SimplePythonDataFrameGraphAdapter, although that's largely for legacy reasons (and can probably be changed).

    TODO -- change this to extend the right class.
    """

    def __init__(self, result_builder: ResultMixin = None):
        """Allows you to swap out the build_result very easily.

        :param result_builder: A ResultMixin object that will be used to build the result.
        """
        if result_builder is None:
            result_builder = DictResult()
        self.result_builder = result_builder

    def build_result(self, **outputs: Dict[str, Any]) -> Any:
        """Delegates to the result builder function supplied."""
        return self.result_builder.build_result(**outputs)

    def output_type(self) -> Type:
        return self.result_builder.output_type()


class DefaultAdapter(SimplePythonGraphAdapter):
    """This is a shortcut for the SimplePythonGraphAdapter. It does the exact same thing,
    but allows for easier access/naming."""
