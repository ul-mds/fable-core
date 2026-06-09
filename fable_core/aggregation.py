__all__ = [
    "AggregationFn",
    "average",
    "maximum",
    "minimum",
]

from typing import Any, Protocol


class AggregationFn(Protocol):
    def __call__(self, similarities: list[float], /, **kwargs: Any) -> float: ...


def _average(similarities: list[float], /, weights: list[float] | None = None) -> float:
    """
    Compute the weighted average of a similarity vector.

    Args:
        similarities: list of similarities
        weights: list of weights

    Returns:
        weighted average of similarities

    Raises:
        ValueError: if length of similarities does not equal length of weights
    """
    if weights is None:
        weights = [1] * len(similarities)
    if len(weights) != len(similarities):
        raise ValueError("There need to be as many weights as there are similarities.")
    return sum(threshold * weight for threshold, weight in zip(similarities, weights, strict=True)) / sum(weights)


# Wrap aggregators so that the type checker does not complain.
def average(similarities: list[float], **kwargs: Any) -> float:
    return _average(similarities, **kwargs)


def maximum(similarities: list[float], **kwargs: Any) -> float:
    return max(similarities, **kwargs)


def minimum(similarities: list[float], **kwargs: Any) -> float:
    return min(similarities, **kwargs)
