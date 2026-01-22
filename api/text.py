from fastapi import APIRouter, Depends
from collections import Counter
import re
import math

from auth import verify_api_key

router = APIRouter(prefix="/text", tags=["TextVortex"])

# ---------------- UTIL ---------------- #
def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

STOPWORDS = {
    "the", "is", "and", "a", "an", "of", "to", "in", "on", "for", "with", "as",
    "by", "at", "from", "that", "this", "it", "are", "was", "were", "be"
}

# ---------------- TOKENIZATION ---------------- #
@router.post("/tokenize")
def tokenize_text(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    return {"tokens": tokens}

# ---------------- STOPWORDS ---------------- #
@router.post("/stopwords")
def remove_stopwords(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    filtered = [t for t in tokens if t not in STOPWORDS]
    return {"tokens": filtered}

# ---------------- STEMMING ---------------- #
@router.post("/stem")
def stem_text(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    stemmed = [t[:-1] if len(t) > 4 else t for t in tokens]
    return {"tokens": stemmed}

# ---------------- LEMMATIZATION ---------------- #
@router.post("/lemmatize")
def lemmatize_text(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    lemmas = [t.rstrip("s") for t in tokens]
    return {"tokens": lemmas}

# ---------------- N-GRAMS ---------------- #
@router.post("/ngrams")
def ngrams(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    n = payload.get("n", 2)
    grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return {"ngrams": grams}

# ---------------- KEYWORDS ---------------- #
@router.post("/keywords")
def keywords(payload: dict, dep=Depends(verify_api_key)):
    tokens = tokenize(payload["text"])
    freq = Counter(tokens)
    return {"keywords": freq}

# ---------------- TEXT STATS ---------------- #
@router.post("/stats")
def stats(payload: dict, dep=Depends(verify_api_key)):
    text = payload["text"]
    words = tokenize(text)
    sentences = re.split(r"[.!?]+", text)
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len([s for s in sentences if s.strip()])
    }

# ---------------- TEXT COMPLEXITY ---------------- #
@router.post("/complexity")
def complexity(payload: dict, dep=Depends(verify_api_key)):
    words = tokenize(payload["text"])
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    return {
        "avg_word_length": round(avg_len, 2),
        "readability_score": round(100 - avg_len * 10, 2)
    }
