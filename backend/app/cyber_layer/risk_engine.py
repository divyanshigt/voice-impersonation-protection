def calculate_overall_risk(
    voice_label: str,
    voice_confidence: float,
    fraud_score: int,
):
    """
    Combines AI voice detection and fraud detection
    into an overall explainable risk assessment.
    """

    # -----------------------------
    # 1. Calculate Voice Risk
    # -----------------------------
    if voice_label.lower() == "spoof":
        voice_risk = voice_confidence * 100
    else:
        voice_risk = (1 - voice_confidence) * 100

    # -----------------------------
    # 2. Fraud Risk
    # -----------------------------
    fraud_risk = fraud_score

    # -----------------------------
    # 3. Overall Risk Score
    # -----------------------------
    overall_score = (
        (voice_risk * 0.6) +
        (fraud_risk * 0.4)
    )

    overall_score = round(overall_score)

    # -----------------------------
    # 4. Risk Level
    # -----------------------------
    if overall_score >= 85:
        risk_level = "CRITICAL"
    elif overall_score >= 70:
        risk_level = "HIGH"
    elif overall_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # -----------------------------
    # 5. Generate Explanation
    # -----------------------------
    reasons = []

    if voice_label.lower() == "spoof":
        reasons.append(
            "AI-generated or manipulated voice detected."
        )

    if voice_confidence >= 0.90 and voice_label.lower() == "spoof":
        reasons.append(
            "Voice detector has very high confidence."
        )

    if fraud_score >= 70:
        reasons.append(
            "Multiple fraud indicators were detected."
        )
    elif fraud_score >= 40:
        reasons.append(
            "Some suspicious fraud indicators were detected."
        )

    # -----------------------------
    # 6. Recommended Action
    # -----------------------------
    if risk_level == "CRITICAL":
        recommended_action = (
            "Do not trust the caller. "
            "Do not share OTP, PIN, password, bank details, "
            "or transfer money."
        )

    elif risk_level == "HIGH":
        recommended_action = (
            "Verify the caller through another trusted channel "
            "before taking any action."
        )

    elif risk_level == "MEDIUM":
        recommended_action = (
            "Proceed with caution and verify the caller's identity."
        )

    else:
        recommended_action = (
            "No major risk indicators detected. "
            "Continue with normal caution."
        )

    return {
        "overall_score": overall_score,
        "risk_level": risk_level,
        "voice_risk": round(voice_risk),
        "fraud_risk": fraud_risk,
        "reasons": reasons,
        "recommended_action": recommended_action,
    }