import pytest

from fable_core import common


@pytest.mark.parametrize(
    "value,expected",
    [
        ("foobar", {"_f", "fo", "oo", "ob", "ba", "ar", "r_"}),
        ("x", {"_x", "x_"}),
        ("", set()),
    ],
)
def test_tokenize_default(value, expected):
    assert common.tokenize(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("foobar", {"#f", "fo", "oo", "ob", "ba", "ar", "r#"}),
        ("x", {"#x", "x#"}),
        ("", set()),
    ],
)
def test_tokenize_with_padding(value, expected):
    assert common.tokenize(value, padding="#") == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("foobar", {"__f", "_fo", "foo", "oob", "oba", "bar", "ar_", "r__"}),
        ("x", {"__x", "_x_", "x__"}),
        ("", set()),
    ],
)
def test_tokenize_with_size(value, expected):
    assert common.tokenize(value, q=3) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("foobar", {"fo", "oo", "ob", "ba", "ar"}),
        ("ab", {"ab"}),
        ("a", set()),
        ("", set()),
    ],
)
def test_tokenize_without_padding(value, expected):
    assert common.tokenize(value, q=2, padding="") == expected


def test_destructure_digest():
    assert common.destructure_digest(b"\x01" * 4 + b"\x23" * 4 + b"\x45" * 4 + b"\x67" * 4) == (
        0x01010101,
        0x23232323,
        0x45454545,
        0x67676767,
    )
