def analyze_fraud(
    voice_label: str,
    voice_confidence: float,
    transcript: str,
):
    """
    Basic fraud-risk analysis using:
    1. AI voice detection result
    2. Voice confidence
    3. Transcript content
    """

    risk_score = 0
    risk_reasons = []

    # -----------------------------
    # 1. AI-generated voice check
    # -----------------------------
    if voice_label.lower() == "spoof":
        risk_score += 60
        risk_reasons.append(
            "AI-generated or manipulated voice detected."
        )

    # -----------------------------
    # 2. Voice confidence
    # -----------------------------
    if voice_confidence >= 0.90 and voice_label.lower() == "spoof":
        risk_score += 20
        risk_reasons.append(
            "High confidence in AI voice detection."
        )

    # -----------------------------
    # 3. Suspicious transcript keywords
    # -----------------------------
    suspicious_keywords = [
        "otp",
        "one time password",
        "password",
        "pin",
        "bank",
        "account",
        "transfer",
        "money",
        "payment",
        "upi",
        "verification",
        "verify",
        "urgent",
        "send",
        "code",
        "cvv",
        "card",
    ]

    transcript_lower = transcript.lower()

    detected_keywords = []

    for keyword in suspicious_keywords:
        if keyword in transcript_lower:
            detected_keywords.append(keyword)

    if detected_keywords:
        risk_score += min(len(detected_keywords) * 5, 20)

        risk_reasons.append(
            "Suspicious financial or sensitive-information "
            "keywords detected."
        )

    # -----------------------------
    # 4. Limit score to 100
    # -----------------------------
    risk_score = min(risk_score, 100)

    # -----------------------------
    # 5. Risk level
    # -----------------------------
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detected_keywords": detected_keywords,
        "reasons": risk_reasons,
    }