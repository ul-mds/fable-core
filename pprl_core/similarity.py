__all__ = [
    "SimilarityFn",
    "dice",
    "cosine",
    "jaccard",
    "russell_rao",
    "sokal_sneath",
    "sokal_michener",
    "roger_tanimoto",
]

import math
from typing import Callable

import bitarray.util as bitarray_util
from bitarray import bitarray

SimilarityFn = Callable[[bitarray, bitarray], float]


def _bitarray_count(ba1: bitarray, ba2: bitarray) -> tuple[int, int, int, int]:
    """
    Count indices where both, only one or none of both bitarrays are set to 1.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Raises:
        ValueError: if the input bitarrays are not the same size

    Returns:
        Number of indices set to 1 in both bitarrays, only in ba2, only in ba1 and none of both

    References:
        Choi et al. (2009). A Survey of Binary Similarity  and Distance Measures.
        https://pdixon.stat.iastate.edu/stat415/Choi.pdf
    """
    a = bitarray_util.count_and(ba1, ba2)
    b = ba2.count() - a
    c = ba1.count() - a
    d = bitarray_util.count_and(~ba1, ~ba2)

    return a, b, c, d


def dice(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using the Dice coefficient.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, _ = _bitarray_count(ba1, ba2)

    return 2 * a / (2 * a + b + c)


def cosine(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using the cosine similarity.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, _ = _bitarray_count(ba1, ba2)

    return a / math.sqrt((a + b) * (a + c))


def jaccard(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using the Jaccard index.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, _ = _bitarray_count(ba1, ba2)

    return a / (a + b + c)


def russell_rao(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using Russell & Rao's similarity.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, d = _bitarray_count(ba1, ba2)

    return a / (a + b + c + d)


def sokal_sneath(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using Sokal & Sneath's similarity.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, _ = _bitarray_count(ba1, ba2)

    return a / (a + 2 * b + 2 * c)


def sokal_michener(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using Sokal & Michener's similarity.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, d = _bitarray_count(ba1, ba2)

    return (a + d) / (a + b + c + d)


def roger_tanimoto(ba1: bitarray, ba2: bitarray) -> float:
    """
    Compute the similarity of two bitarrays using Roger & Tanimoto's similarity.

    Args:
        ba1: first bitarray
        ba2: second bitarray

    Returns:
        similarity of bitarrays
    """
    a, b, c, d = _bitarray_count(ba1, ba2)

    return (a + d) / (a + 2 * b + 2 * c + d)
