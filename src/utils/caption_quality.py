"""Shared quality gates for generated audio captions."""

import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z']+")
BAD_PATTERN_RE = re.compile(
    "|".join(
        [
            r"\b[a-z]?aa+\b",
            r"\b[a-z]*(?:ss|sss|ringring|birdsa|birdsas|blowss|pocking|speaksa|plupp|slopp)[a-z]*\b",
            r"\b\w{18,}\b",
            r"sss+|ssr|shsk|squiw|jerss|splashjerp|suggs|squecks|snicks|witha|severaling",
            r"skiping|inginging|sprinks|anding|birdss|flawag|hummings|gauning|swooing|wries",
            r"splashs|runss|quacks?|chir\s*$|followed\s*bya|watera|anda|food\s+is\s*$",
            r"\ba\s+(\w+),?\s+a\s+\1\b",
            r"\b(\w+)s?\s+\1s?\b",
            r"\b(\w+)\s+and\s+\1\b",
            r"\b(?:followed\s+by|then)\s+a\s+(?:man|woman|person|vehicle)\s*$",
            r"\b(?:followed\s+by|then)\s+a\s+(?:man|woman|person|vehicle|bird|cat|dog)\b",
            r"\band\s+a\s+and\b",
            r"\band\s+a\s+(?:blow|laugh|speaks?|talks?)\b",
            r"\ba\s+(?:water|wind|rain)\b",
            r"\bthen\s+and\b",
            r"\b(?:makes|make)\s+laughing\b",
            r"\ba\s+(?:speaks|talks|laughs|cries|barks|chirps|blows|runs|idles)\b",
            r"\b(?:speaks|talks)\s+followed\s+by\s+a\s+(?:bird|cat|dog)\b",
            r"\bby\s+as\b",
            r"\bcarhorn\b",
            r"\bwaves\s+blows\b",
            r"\b\w*ows\b",
            r"\b\w*waters\b",
            r"\b\w*rings\b",
            r"\b\w*(?:zzle|jering|yellly|whi)\w*\b",
            r"\b\w*ing(?:ing)+\w*\b",
        ]
    ),
    re.IGNORECASE,
)

BAD_TERMINALS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "followed",
    "is",
    "laugh",
    "man",
    "of",
    "person",
    "the",
    "then",
    "to",
    "vehicle",
    "woman",
    "with",
}

ALLOWED_SHORT_TOKENS = {"a", "i", "an", "as", "by", "in", "is", "of", "on", "or", "to"}
ALLOWED_ING_TOKENS = {
    "barking",
    "beeping",
    "blowing",
    "chirping",
    "clapping",
    "clicking",
    "coughing",
    "crying",
    "dripping",
    "falling",
    "flowing",
    "honking",
    "laughing",
    "meowing",
    "pouring",
    "raining",
    "ringing",
    "rolling",
    "running",
    "singing",
    "speaking",
    "splashing",
    "talking",
    "ticking",
    "tweeting",
    "walking",
    "whistling",
}

BAD_WHOLE_TOKENS = {
    "anda",
    "bya",
    "chir",
    "guns",
    "carhorn",
    "splashs",
    "watera",
    "ows",
    "raina",
    "snor",
}


@dataclass(frozen=True)
class CaptionQuality:
    ok: bool
    reason: str | None
    num_tokens: int
    adjacent_repeats: int


def caption_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def adjacent_repeat_count(tokens: list[str]) -> int:
    return sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)


def bad_terminal(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[-1] in BAD_TERMINALS


def bad_token(tokens: list[str]) -> bool:
    for tok in tokens:
        if len(tok) <= 2 and tok not in ALLOWED_SHORT_TOKENS:
            return True
        if tok in BAD_WHOLE_TOKENS:
            return True
        if _has_internal_repeat(tok):
            return True
        if len(tok) >= 5 and tok.endswith("a") and tok not in {"camera", "salsa", "opera", "tuba"}:
            return True
        if len(tok) > 3 and (tok.endswith("ss") or tok.endswith("sss")):
            return True
        if "shot" in tok and tok != "shot":
            return True
        if tok in {"someone", "another"}:
            return True
        if tok.endswith("ing") and tok not in ALLOWED_ING_TOKENS:
            return True
    return False


def has_phrase_repeat(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    for idx in range(0, len(tokens) - 2):
        if tokens[idx + 1] == "and" and tokens[idx] == tokens[idx + 2]:
            return True
    for width in (2, 3):
        for start in range(0, len(tokens) - 2 * width + 1):
            if tokens[start : start + width] == tokens[start + width : start + 2 * width]:
                return True
            end = start + width
            if (
                end < len(tokens)
                and tokens[end] == "and"
                and end + 1 + width <= len(tokens)
                and tokens[start:end] == tokens[end + 1 : end + 1 + width]
            ):
                return True
    return False


def _has_internal_repeat(token: str) -> bool:
    if len(token) < 6:
        return False
    for width in range(2, max(2, len(token) // 2) + 1):
        for start in range(0, len(token) - 2 * width + 1):
            chunk = token[start : start + width]
            if chunk == token[start + width : start + 2 * width]:
                return True
    return False


def assess_caption_quality(
    text: str,
    min_tokens: int = 3,
    max_tokens: int = 18,
    allow_adjacent_repeat: bool = False,
    reject_bad_token: bool = True,
) -> CaptionQuality:
    tokens = caption_tokens(text)
    repeats = adjacent_repeat_count(tokens)
    if not text or not text.strip():
        return CaptionQuality(False, "empty", len(tokens), repeats)
    if len(tokens) < min_tokens:
        return CaptionQuality(False, "too_short", len(tokens), repeats)
    if len(tokens) > max_tokens:
        return CaptionQuality(False, "too_long", len(tokens), repeats)
    if BAD_PATTERN_RE.search(text or ""):
        return CaptionQuality(False, "bad_pattern", len(tokens), repeats)
    if bad_terminal(tokens):
        return CaptionQuality(False, "bad_terminal", len(tokens), repeats)
    if repeats > 0 and not allow_adjacent_repeat:
        return CaptionQuality(False, "adjacent_repeat", len(tokens), repeats)
    if has_phrase_repeat(tokens):
        return CaptionQuality(False, "phrase_repeat", len(tokens), repeats)
    if reject_bad_token and bad_token(tokens):
        return CaptionQuality(False, "bad_token", len(tokens), repeats)
    return CaptionQuality(True, None, len(tokens), repeats)
