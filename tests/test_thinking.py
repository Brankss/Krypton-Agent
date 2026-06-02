"""Chain-of-thought stripping + reasoning-effort normalization.

`strip_think` must survive a <think> tag being split across streaming chunk
boundaries (DeepSeek / Nemotron / Qwen3 inline their reasoning in the content
channel). This is the subtle part of the providers, so it is locked in here.
"""
from __future__ import annotations

import pytest

from krypton.providers._thinking import normalize_effort, strip_think


def _run(chunks):
    out, in_think, carry = [], False, ""
    for ch in chunks:
        clean, in_think, carry = strip_think(ch, in_think, carry)
        out.append(clean)
    return "".join(out), in_think, carry


def test_strip_single_chunk():
    vis, in_think, _ = _run(["before<think>hidden</think>after"])
    assert vis == "beforeafter"
    assert in_think is False


def test_strip_tag_split_across_chunks():
    vis, in_think, _ = _run(["abc<thi", "nk>secret</thi", "nk>xyz"])
    assert "secret" not in vis
    assert vis == "abcxyz"
    assert in_think is False


def test_plain_text_passthrough():
    vis, _, _ = _run(["just plain ", "text here"])
    assert vis == "just plain text here"


def test_unterminated_think_holds_state():
    vis, in_think, _ = _run(["visible<think>still reasoning"])
    assert vis == "visible"
    assert in_think is True


def test_multiple_think_blocks():
    vis, _, _ = _run(["a<think>x</think>b<think>y</think>c"])
    assert vis == "abc"


def test_normalize_effort_valid():
    assert normalize_effort("LOW") == "low"
    assert normalize_effort(" High ") == "high"
    assert normalize_effort("medium") == "medium"
    assert normalize_effort("none") == "none"


def test_normalize_effort_empty_is_none():
    assert normalize_effort("") is None
    assert normalize_effort(None) is None


def test_normalize_effort_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_effort("turbo")
