"""Name normalisation and aliases for matching people in video titles."""

from __future__ import annotations

import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize(text: str) -> str:
    """Lowercase ASCII words separated by single spaces ("Rajaković" -> "rajakovic")."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"'", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def last_name(name: str) -> str:
    tokens = [t for t in normalize(name).split() if t not in SUFFIXES]
    return tokens[-1] if tokens else ""


def full_forms(name: str) -> list[str]:
    """Normalised spellings of a full name: "J. B. Bickerstaff" -> j b / jb ..."""
    tokens = normalize(name).split()
    if not tokens:
        return []
    forms = {" ".join(tokens)}
    initials = 0
    while initials < len(tokens) - 1 and len(tokens[initials]) == 1:
        initials += 1
    if initials >= 2:
        forms.add(" ".join(["".join(tokens[:initials]), *tokens[initials:]]))
    without_suffix = [t for t in tokens if t not in SUFFIXES]
    if len(without_suffix) != len(tokens) and len(without_suffix) > 1:
        forms.add(" ".join(without_suffix))
    return sorted(forms)


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word containment on normalised strings."""
    return bool(phrase) and f" {phrase} " in f" {text} "
