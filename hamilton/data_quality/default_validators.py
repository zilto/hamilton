import logging
import numbers
from typing import Any, Iterable, List, Tuple, Type, Union

from hamilton.data_quality import base

logger = logging.getLogger(__name__)


class DataInRangeValidatorPrimitives(base.BaseDefaultValidator):
    def __init__(self, range: Tuple[numbers.Real, numbers.Real], importance: str):
        """Data validator that tells if data is in a range. This applies to primitives (ints, floats).

        :param range: Inclusive range of parameters
        """
        super(DataInRangeValidatorPrimitives, self).__init__(importance=importance)
        self.range = range

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return issubclass(datatype, numbers.Real)

    def description(self) -> str:
        return f"Validates that the datapoint falls within the range ({self.range[0]}, {self.range[1]})"

    def validate(self, data: numbers.Real) -> base.ValidationResult:
        min_, max_ = self.range
        if hasattr(data, "dask"):
            data = data.compute()
        passes = min_ <= data <= max_
        if passes:
            message = f"Data point {data} falls within acceptable range: ({min_}, {max_})"
        else:
            message = f"Data point {data} does not fall within acceptable range: ({min_}, {max_})"
        return base.ValidationResult(
            passes=passes,
            message=message,
            diagnostics={"range": self.range, "value": data},
        )

    @classmethod
    def arg(cls) -> str:
        return "range"


class DataInValuesValidatorPrimitives(base.BaseDefaultValidator):
    def __init__(self, values_in: Iterable[Any], importance: str):
        """Data validator that tells if python primitive type data is in a set of specified values.

        :param values_in: list of valid values
        """
        super(DataInValuesValidatorPrimitives, self).__init__(importance=importance)
        self.values = frozenset(values_in)

    @classmethod
    def arg(cls) -> str:
        return "values_in"

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return issubclass(datatype, numbers.Real) or issubclass(
            datatype, str
        )  # TODO support list, dict and typing.* variants

    def description(self) -> str:
        return f"Validates that python values are from a fixed set of values: ({self.values})."

    def validate(self, data: Union[numbers.Real, str]) -> base.ValidationResult:
        if hasattr(data, "dask"):
            data = data.compute()
        is_valid_value = data in self.values
        message = f"Primitive python value was valid is {is_valid_value}."
        if not is_valid_value:
            message += f" Correct possible values are {self.values}."
        return base.ValidationResult(
            passes=is_valid_value,
            message=message,
            diagnostics={
                "values": self.values,
                "was_correct": is_valid_value,
                "incorrect_value": None if is_valid_value else data,
                "data_size": 1,
            },
        )


class DataTypeValidatorPrimitives(base.BaseDefaultValidator):
    def __init__(self, data_type: Type[Type], importance: str):
        """Constructor

        :param data_type: the python data type to expect.
        """
        super(DataTypeValidatorPrimitives, self).__init__(importance=importance)
        DataTypeValidatorPrimitives.datatype = data_type
        self.datatype = data_type

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return issubclass(datatype, numbers.Real) or datatype in (str, bool)

    def description(self) -> str:
        return f"Validates that the datatype of the pandas series is a subclass of: {self.datatype}"

    def validate(
        self, data: Union[numbers.Real, str, bool, int, float, list, dict]
    ) -> base.ValidationResult:
        if hasattr(data, "dask"):
            data = data.compute()
        passes = isinstance(data, self.datatype)
        return base.ValidationResult(
            passes=passes,
            message=f"Requires data type: {self.datatype}. "
            f"Got data type: {type(data)}. This {'is' if passes else 'is not'} a match.",
            diagnostics={
                "required_data_type": self.datatype,
                "actual_data_type": type(data),
            },
        )

    @classmethod
    def arg(cls) -> str:
        return "data_type"


class AllowNoneValidator(base.BaseDefaultValidator):
    def __init__(self, allow_none: bool, importance: str):
        super(AllowNoneValidator, self).__init__(importance)
        self.allow_none = allow_none

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return True

    def description(self) -> str:
        if self.allow_none:
            return "No-op validator."
        return "Validates that an output ;is not None"

    def validate(self, data: Any) -> base.ValidationResult:
        passes = True
        if not self.allow_none:
            if data is None:
                passes = False
        return base.ValidationResult(
            passes=passes,
            message=(
                f"Data is not allowed to be None, got {data}" if not passes else "Data is not None"
            ),
            diagnostics={},  # Nothing necessary here...
        )

    @classmethod
    def arg(cls) -> str:
        return "allow_none"


class StrContainsValidator(base.BaseDefaultValidator):
    def __init__(self, contains: Union[str, List[str]], importance: str):
        super(StrContainsValidator, self).__init__(importance)
        if isinstance(contains, str):
            self.contains = [contains]
        else:
            self.contains = contains

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return datatype == str

    def description(self) -> str:
        return f"Validates that a string contains [{self.contains}] within it."

    def validate(self, data: str) -> base.ValidationResult:
        passes = all([c in data for c in self.contains])
        return base.ValidationResult(
            passes=passes,
            message=(f"String did not contain {self.contains}" if not passes else "All good."),
            diagnostics=(
                {"contains": self.contains, "data": data if len(data) < 100 else data[:100]}
                if not passes
                else {}
            ),
        )

    @classmethod
    def arg(cls) -> str:
        return "contains"


class StrDoesNotContainValidator(base.BaseDefaultValidator):
    def __init__(self, does_not_contain: Union[str, List[str]], importance: str):
        super(StrDoesNotContainValidator, self).__init__(importance)
        if isinstance(does_not_contain, str):
            self.does_not_contain = [does_not_contain]
        else:
            self.does_not_contain = does_not_contain

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return datatype == str

    def description(self) -> str:
        return f"Validates that a string does not contain [{self.does_not_contain}] within it."

    def validate(self, data: str) -> base.ValidationResult:
        passes = all([c not in data for c in self.does_not_contain])
        return base.ValidationResult(
            passes=passes,
            message=(f"String did contain {self.does_not_contain}" if not passes else "All good."),
            diagnostics=(
                {
                    "does_not_contain": self.does_not_contain,
                    "data": data if len(data) < 100 else data[:100],
                }
                if not passes
                else {}
            ),
        )

    @classmethod
    def arg(cls) -> str:
        return "does_not_contain"


AVAILABLE_DEFAULT_VALIDATORS = [
    DataInRangeValidatorPrimitives,
    DataInValuesValidatorPrimitives,
    DataTypeValidatorPrimitives,
    AllowNoneValidator,
    StrContainsValidator,
    StrDoesNotContainValidator,
]


def _append_pandera_to_default_validators():
    """Utility method to append pandera validators as needed"""
    try:
        import pandera  # noqa: F401
    except ModuleNotFoundError:
        logger.info(
            "Cannot import pandera from pandera_validators. Run pip install sf-hamilton[pandera] if needed."
        )
        return
    from hamilton.data_quality import pandera_validators

    AVAILABLE_DEFAULT_VALIDATORS.extend(pandera_validators.PANDERA_VALIDATORS)


_append_pandera_to_default_validators()


def resolve_default_validators(
    output_type: Type[Type],
    importance: str,
    available_validators: List[Type[base.BaseDefaultValidator]] = None,
    **default_validator_kwargs,
) -> List[base.BaseDefaultValidator]:
    """Resolves default validators given a set pof parameters and the type to which they apply.
    Note that each (kwarg, type) combination should map to a validator
    :param importance: importance level of the validator to instantiate
    :param output_type: The type to which the validator should apply
    :param available_validators: The available validators to choose from
    :param default_validator_kwargs: Kwargs to use
    :return: A list of validators to use
    """
    if available_validators is None:
        available_validators = AVAILABLE_DEFAULT_VALIDATORS
    validators = []
    for key in default_validator_kwargs.keys():
        for validator_cls in available_validators:
            if key == validator_cls.arg() and validator_cls.applies_to(output_type):
                validators.append(
                    validator_cls(**{key: default_validator_kwargs[key], "importance": importance})
                )
                break
        else:
            raise ValueError(
                f"No registered subclass of BaseDefaultValidator is available "
                f"for arg: {key} and type {output_type}. This either means (a) this arg-type "
                f"contribution isn't supported or (b) this has not been added yet (but should be). "
                f"In the case of (b), we welcome contributions. Get started at github.com/dagworks-inc/hamilton."
            )
    return validators
