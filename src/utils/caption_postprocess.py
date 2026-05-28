import re


_ARTIFACT_PATTERNS = (
    r"\b\w*(?:zzle|jering|yellly|whi)\w*\b",
    r"\b\w*ing(?:ing)+\w*\b",
    r"\b\w*(?:ffle|ggle)\w*\b",
)


def clean_caption_artifacts(text: str) -> str:
    """Remove common DLM decoder artifacts without adding semantic content."""
    text = text.replace("\u2581", " ")
    for pattern in _ARTIFACT_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z']+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:")
    text = re.sub(r"\b(and|then|followed by)\s*$", "", text, flags=re.IGNORECASE).strip(" ,;:")
    return text
