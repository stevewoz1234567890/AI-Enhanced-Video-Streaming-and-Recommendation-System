"""Lexical + TF–IDF style theme scoring over dialogue / transcript text (swap in transformer NLP as needed)."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Seed phrases per theme — expand from labeled data or fine-tuned embeddings.
THEME_SEEDS: dict[str, list[str]] = {
    "action": [
        "fight chase explosion battle weapon shoot crash escape pursuit combat",
        "showdown ambush detonation pursuit vehicle stunt",
    ],
    "romance": [
        "love kiss relationship wedding heart break date couple affection",
        "marriage proposal intimate feelings passion",
    ],
    "horror": [
        "fear dark blood nightmare ghost demon haunted scream terror",
        "monster curse grave supernatural dread",
    ],
    "comedy": [
        "joke laugh humor funny prank awkward silly comedy banter",
        "ridiculous absurd hilarious",
    ],
    "drama": [
        "family secret betrayal loss grief choice consequence tension",
        "struggle reconciliation confession",
    ],
    "sci_fi": [
        "space ship robot future technology planet artificial intelligence",
        "laser colony terraform android",
    ],
}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 12]


def score_themes(transcript: str) -> list[tuple[str, float]]:
    sents = _sentences(transcript)
    if not sents:
        return [(k, 0.0) for k in THEME_SEEDS]

    corpus = sents + [p for phrases in THEME_SEEDS.values() for p in phrases]
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(corpus)
    X_sents = X[: len(sents)]
    start = len(sents)
    theme_vecs = {}
    for theme, phrases in THEME_SEEDS.items():
        block = X[start : start + len(phrases)]
        start += len(phrases)
        theme_vecs[theme] = block.mean(axis=0)

    scores: dict[str, float] = {}
    for theme, tv in theme_vecs.items():
        sims = (X_sents @ tv.T).toarray().ravel()
        scores[theme] = float(sims.max()) if sims.size else 0.0

    m = max(scores.values()) if scores else 1.0
    if m <= 1e-9:
        return sorted(((k, 0.0) for k in scores), key=lambda x: x[0])
    return sorted(((k, v / m) for k, v in scores.items()), key=lambda x: -x[1])
