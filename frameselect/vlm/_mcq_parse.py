# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared MCQ-letter extractor for every VLM backend.

The parser recognises the common shapes VLMs return when asked for a
single option letter:

* bare letter: ``"A"``, ``"B."``, ``"(C)"``
* labelled: ``"Answer: B"``, ``"Answer is (C)"``
* prose + letter: ``"The answer is B."``

Critically, the final "prose + letter" branch anchors on a **word
boundary at the end** so ``"The answer is B."`` picks ``"B"`` rather
than the ``"A"`` hidden inside ``"answer"``; a naive
``re.search(r"[A-D]", text.upper())`` would pick the latter whenever a
model replies with more than one token.

Option-count awareness
----------------------
Benchmarks are not all 4-way. ActivityNet has 5 options
(``A..E``), and other MCQ benchmarks have up to 7 (``A..G``). The parser therefore
accepts an ``n_options`` argument and widens its valid-letter range to
``A..chr(ord('A')+n_options-1)``. When ``n_options`` is unknown it
falls back to the historical ``A..D`` window. Backends thread the real
option count through :func:`build_mcq_message` and their ``call_*_row``
wrappers so 5--7-way questions can actually return ``E``/``F``/``G``.

Failure sentinel
----------------
:data:`FAILED_ANSWER` is the value backends return when a VLM call
genuinely fails (all retries exhausted, no endpoints, hard error). It
is deliberately **not** a real option letter so downstream evaluation
can distinguish a failed call from a real answer and avoid silently
biasing accuracy toward ``"A"``.
"""

from __future__ import annotations

import re
from typing import Optional

# Sentinel returned by backends on a genuine call failure (exhausted
# retries / no endpoints / hard error). NOT a valid option letter, so
# evaluations can count it wrong/excluded instead of silently scoring
# it as "A". Importable by the evaluations layer.
FAILED_ANSWER = ""

# Fallback option count when the caller doesn't know how many options a
# question has. Preserves the historical A-D behaviour.
_DEFAULT_N_OPTIONS = 4

# MCQ letters never exceed the alphabet.
_MAX_N_OPTIONS = 26


def _clamp_n_options(n_options: Optional[int]) -> int:
    """Resolve a usable option count, falling back to A-D when unknown."""
    if n_options is None or n_options < 1:
        return _DEFAULT_N_OPTIONS
    return min(n_options, _MAX_N_OPTIONS)


def _max_letter(n_options: Optional[int]) -> str:
    """Highest valid option letter for ``n_options`` (e.g. 5 -> ``"E"``)."""
    return chr(ord("A") + _clamp_n_options(n_options) - 1)


def option_letters(n_options: Optional[int]) -> list[str]:
    """Return the list of valid option letters, e.g. ``["A","B","C","D","E"]``.

    Used by the OpenAI backend to build a ``guided_choice`` constraint
    that matches the real number of options.
    """
    n = _clamp_n_options(n_options)
    return [chr(ord("A") + i) for i in range(n)]


def format_letter_phrase(n_options: Optional[int]) -> str:
    """Human-readable list of the valid letters for a prompt.

    ``2 -> "A or B"``, ``4 -> "A, B, C, or D"``,
    ``7 -> "A, B, C, D, E, F, or G"``.
    """
    letters = option_letters(n_options)
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def mcq_answer_instruction(n_options: Optional[int]) -> str:
    """Full 'answer with only the letter' instruction line for prompts."""
    return f"Answer with ONLY the option letter ({format_letter_phrase(n_options)})."


def parse_mcq_letter(
    text: str,
    default: str = "A",
    n_options: Optional[int] = None,
) -> str:
    """Extract an MCQ answer letter from a VLM response.

    Parameters
    ----------
    text : str
        Raw text from the VLM. Case-insensitive; surrounding
        whitespace and punctuation are tolerated.
    default : str
        Letter to return when no match is found. Defaults to ``"A"``,
        matching the historical behaviour of the per-backend parsers.
    n_options : int, optional
        Number of options the question offers. Widens the valid-letter
        range to ``A..chr(ord('A')+n_options-1)`` so 5--7-way questions
        can return ``E``/``F``/``G``. When ``None`` (unknown) the range
        falls back to ``A..D``.

    Returns
    -------
    str
        A single option letter within the valid range, or ``default``.
    """
    if not text:
        return default
    maxl = _max_letter(n_options)
    cls = f"[A-{maxl}]"
    upper = text.upper().strip()

    # 1. Bare / wrapped letter with optional trailing punctuation.
    #    "A", "A.", "(A)", "A)".
    m = re.match(rf"^\s*\(?\s*({cls})\s*[\).:\-]?\s*$", upper)
    if m:
        return m.group(1)

    # 2. Labelled form: "Answer: B" / "Answer is C".
    #    Requires the literal "ANSWER" keyword so prose sentences
    #    starting with a single-letter word aren't misclassified.
    m = re.match(rf"^\s*ANSWER(?:\s+IS)?[:\s]*\(?\s*({cls})\b", upper)
    if m:
        return m.group(1)

    # 3. The LAST word-boundaried valid letter not immediately followed
    #    by another word character. Critical for prose like
    #    "Considering A, B, C — the answer is D." where picking the
    #    first boundaried letter would grab "A" instead of "D".
    matches = list(re.finditer(rf"\b({cls})\b(?!\s*\w)", upper))
    if matches:
        return matches[-1].group(1)

    # 4. Any word-boundaried valid letter anywhere in the string.
    m = re.search(rf"\b({cls})\b", upper)
    if m:
        return m.group(1)

    return default
