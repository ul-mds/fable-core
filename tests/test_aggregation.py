import pytest

from fable_core import aggregation


@pytest.mark.parametrize(
    "similarities,weights,expected",
    [
        ([0.6, 0.3, 0.6], None, 0.5),
        ([0.6, 0.3, 0.6], [1, 1, 1], 0.5),
        ([0.5, 0.8], [2, 1], 0.6),
    ],
)
def test_average_aggregation(similarities, weights, expected):
    assert aggregation.average(similarities, weights=weights) == expected


def test_average_value_error():
    with pytest.raises(ValueError) as e:
        aggregation.average([0.5, 0.7, 0.8], weights=[2, 3.2])

    assert "There need to be as many weights as there are similarities." in str(e.value)
