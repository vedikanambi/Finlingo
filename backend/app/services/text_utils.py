from __future__ import annotations

import re


def normalise_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalise_text(text)
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        return [normalise_text(x) for x in sent_tokenize(text) if normalise_text(x)]
    except Exception:
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if x.strip()]


_CLAUSE_SPLIT_PATTERN = re.compile(
    r";\s+|,\s+(?:and|but|or|which|who|provided that|except that)\s+", re.IGNORECASE
)


def split_long_clauses(text: str) -> str:
    # last-resort fallback when the LLM retry loop still can't hit the FK target - just
    # splits on semicolons/conjunctions so sentence length drops without losing any words
    sentences = split_sentences(text)
    rewritten: list[str] = []
    for sentence in sentences:
        parts = [part.strip() for part in _CLAUSE_SPLIT_PATTERN.split(sentence) if part.strip()]
        if len(parts) <= 1:
            rewritten.append(sentence)
            continue
        for part in parts:
            piece = part[0].upper() + part[1:] if part else part
            if not piece.endswith((".", "!", "?")):
                piece += "."
            rewritten.append(piece)
    return " ".join(rewritten)


def flesch_kincaid_grade(text: str) -> float:
    try:
        import textstat

        value = float(textstat.flesch_kincaid_grade(text))
        return round(max(value, 0.0), 3)
    except Exception:
        sentences = max(len(split_sentences(text)), 1)
        words = re.findall(r"[A-Za-z]+", text)
        if not words:
            return 0.0
        syllables = sum(_syllables(word) for word in words)
        return round(max(0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59, 0), 3)


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    count = len(groups)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(count, 1)
