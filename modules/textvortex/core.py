def sentiment_analysis(text: str):
    text = text.lower()
    if "good" in text or "excellent" in text:
        return "Positive"
    elif "bad" in text or "poor" in text:
        return "Negative"
    return "Neutral"
