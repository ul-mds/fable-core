import pytest
from bitarray import bitarray

from pprl_core import similarity

_ba_left = bitarray("1" * 40 + "0" * 10)
_ba_right = bitarray("1" * 10 + "0" * 35 + "1" * 5)


def test_bitarray_count():
    assert sum(similarity._bitarray_count(_ba_left, _ba_right)) == len(_ba_left) == len(_ba_right)
    assert similarity._bitarray_count(_ba_left, _ba_right) == (10, 5, 30, 5)


@pytest.mark.parametrize(
    "measure,expected",
    [
        (similarity.dice, 0.3636),
        (similarity.cosine, 0.4082),
        (similarity.jaccard, 0.2222),
        (similarity.russell_rao, 0.2),
        (similarity.sokal_sneath, 0.125),
        (similarity.sokal_michener, 0.3),
        (similarity.roger_tanimoto, 0.1765),
    ],
)
def test_similarity_measures(measure, expected):
    assert pytest.approx(measure(_ba_left, _ba_right), abs=1e-4) == expected
