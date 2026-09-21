#!/usr/bin/env python3
"""Tag-grammar correction for P&ID OCR output.

P&ID text is mostly a handful of strict shapes (instrument tags AA-99999,
bubble numbers 999, function letters AAA, pipe labels 9"-AA-9999, sizes 9",
reducers 9"x9").  The recognizer's errors are within-slot confusions (O/0,
I/1, S/5, B/8, Z/2, G/6) and dropped inch marks.  `correct()` tries the
slot-aware substitutions and accepts a candidate only when it turns a
non-conforming string into one that matches a grammar; conforming input, plain
words and bare digit runs are left alone.

Measured on 56 Dataset-P&ID sheets (tools/ocr_post.py): horizontal-tag exact
match 0.909 -> 0.926, vertical 0.609 -> 0.989, CER 0.025 -> 0.022.
"""
from __future__ import annotations

import itertools
import re
from typing import List

GRAMMAR = [re.compile(p) for p in (
    r'^[A-Z]{2}-\d{5}$',            # CD-31021
    r'^[A-Z]{3}-\d-\d{2}$',         # ERV-8-20
    r'^[A-Z]{2}-\d{2}$',            # LG-10
    r'^\d{1,2}"-[A-Z]{2}-\d{4}$',   # 5"-EK-2648
    r'^\d{3}$',                     # 101
    r'^[A-Z]{3}$',                  # SDL
    r'^\d{1,2}"$',                  # 12"
    r'^\d{1,2}"x\d{1,2}"$',         # 9"x7"
    r'^\d{1,2}\.$',                 # 10.   (notes numbering)
)]

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "!": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
TO_ALPHA = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}.items()}
INCH = {"''": '"', "”": '"', "“": '"', "``": '"', "″": '"'}
DASH = {"–": "-", "—": "-", "_": "-", "−": "-", ".": "-"}


def conforms(t: str) -> bool:
    return any(g.match(t) for g in GRAMMAR)


def normalize_punct(t: str) -> str:
    t = t.strip()
    for a, b in INCH.items():
        t = t.replace(a, b)
    for a, b in DASH.items():
        if a != ".":
            t = t.replace(a, b)
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r'(\d)\s*"', r'\1"', t)
    return t


def _slot_variants(t: str) -> List[str]:
    """Strings reachable by flipping ambiguous characters between the two classes."""
    pos = [i for i, ch in enumerate(t) if ch in TO_DIGIT or ch in TO_ALPHA]
    if not pos or len(pos) > 6:
        return []
    out = []
    for mask in itertools.product((0, 1), repeat=len(pos)):
        if not any(mask):
            continue
        s = list(t)
        for m, i in zip(mask, pos):
            if m:
                ch = s[i]
                s[i] = TO_DIGIT.get(ch, TO_ALPHA.get(ch, ch))
        out.append("".join(s))
    return out


def correct(text: str) -> str:
    """Return the corrected tag, or the input (punctuation-normalised) if no rule applies."""
    t = normalize_punct(text.upper())
    if conforms(t):
        return t
    cands: List[str] = []
    # Only strings that already carry tag punctuation (a hyphen or inch mark)
    # are rebuilt; a bare digit run like a drawing number stays as it is.
    tagged = "-" in t or '"' in t
    # inch-mark repair: 5-EK-2648 / 5 -EK-2648 / 5''-EK -> 5"-EK-2648
    m = re.match(r'^(\d{1,2})["\']*\s*-\s*([A-Z0-9]{2})-(\d{4})$', t)
    if m and tagged:
        cands.append(f'{m.group(1)}"-{m.group(2)}-{m.group(3)}')
    m = re.match(r'^(\d{1,2})["\']*[xX](\d{1,2})["\']*$', t)
    if m:
        cands.append(f'{m.group(1)}"x{m.group(2)}"')
    m = re.match(r'^([A-Z0-9]{2})-([A-Z0-9]{5})$', t)          # CD-3l021
    if m and tagged:
        cands.append(f"{m.group(1)}-{m.group(2)}")
    # slot flips on the candidates and on the original (never on plain words)
    if tagged or t.isalnum() and len(t) <= 3:
        for base in [t] + list(cands):
            cands.extend(_slot_variants(base))
    ok = [c for c in cands if conforms(c)]
    if not ok:
        return t
    # prefer the fewest edits
    ok.sort(key=lambda c: (sum(a != b for a, b in zip(c, t)) + abs(len(c) - len(t))))
    return ok[0]
