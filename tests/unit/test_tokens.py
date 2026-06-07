from __future__ import annotations

import pytest

from agentcost.tokens import CHARS_PER_TOKEN, HeuristicTokenizer, Tokenizer, estimate_tokens


def test_estimate_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_min_one_token():
    assert estimate_tokens("a") == 1
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 1


def test_estimate_ceiling_division():
    # 5 chars / 4 = ceil(1.25) = 2
    assert estimate_tokens("abcde") == 2
    # 8 chars / 4 = 2
    assert estimate_tokens("a" * 8) == 2
    # 9 chars / 4 = ceil(2.25) = 3
    assert estimate_tokens("a" * 9) == 3


def test_estimate_large():
    assert estimate_tokens("x" * 4000) == 1000


def test_estimate_custom_chars_per_token():
    assert estimate_tokens("a" * 10, chars_per_token=5) == 2
    assert estimate_tokens("a" * 11, chars_per_token=5) == 3


def test_estimate_invalid_chars_per_token():
    with pytest.raises(ValueError):
        estimate_tokens("hello", chars_per_token=0)
    with pytest.raises(ValueError):
        estimate_tokens("hello", chars_per_token=-1)


def test_default_chars_per_token_constant():
    assert CHARS_PER_TOKEN == 4


def test_heuristic_tokenizer_matches_function():
    tok = HeuristicTokenizer()
    assert tok.count("a" * 9) == estimate_tokens("a" * 9)


def test_heuristic_tokenizer_custom():
    tok = HeuristicTokenizer(chars_per_token=2)
    assert tok.count("abcd") == 2


def test_heuristic_tokenizer_invalid():
    with pytest.raises(ValueError):
        HeuristicTokenizer(chars_per_token=0)


def test_heuristic_satisfies_protocol():
    tok = HeuristicTokenizer()
    assert isinstance(tok, Tokenizer)


def test_custom_tokenizer_protocol():
    class WordTokenizer:
        def count(self, text: str) -> int:
            return len(text.split())

    tok = WordTokenizer()
    assert isinstance(tok, Tokenizer)
    assert tok.count("one two three") == 3
