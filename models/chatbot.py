def chatbot_reply(user_msg: str) -> str:
    if not user_msg:
        return (
            "Hello — I can help with general health guidance. "
            "Tell me your symptoms, or try the image or voice analysis tools. "
            "This system does not provide medical diagnoses."
        )

    msg = user_msg.lower()

    # Red flag / urgent symptoms
    urgent_signs = [
        "chest pain", "severe chest", "crushing chest",
        "shortness of breath", "hard to breathe", "can't breathe",
        "fainting", "passed out",
        "stroke", "face drooping", "slurred speech",
        "seizure",
        "bleeding a lot", "heavy bleeding",
        "head injury", "loss of consciousness",
        "high fever", "fever 40", "fever 104"
    ]

    if any(term in msg for term in urgent_signs):
        return (
            "Some of the symptoms you described may indicate a medical emergency. "
            "I cannot diagnose conditions, but it would be safest to seek urgent medical attention "
            "or contact emergency services now. If you are unsure, err on the side of caution."
        )

    # Respiratory symptoms
    if "cough" in msg or "breath" in msg or "wheezing" in msg:
        return (
            "Cough and breathing symptoms can come from infections, allergies, or irritation. "
            "Monitor your breathing, stay hydrated, and avoid smoke/irritants. "
            "Seek medical advice if symptoms worsen, last more than a few days, "
            "or you develop chest pain or fever. "
            "You may also try the voice tool — it can help flag possible vocal or breathing anomalies, "
            "but it is not diagnostic."
        )

    # Fever / infection-like symptoms
    if "fever" in msg or "temperature" in msg or "flu" in msg:
        return (
            "Fever is often the body’s response to infection. Rest, fluids, and monitoring can help. "
            "Seek medical advice if the fever persists beyond three days, rises very high, "
            "or if you experience confusion, severe weakness, or persistent chest pain. "
            "I cannot diagnose illness, but I can help you think about symptoms."
        )

    # Skin / rash
    if "rash" in msg or "skin" in msg or "spots" in msg:
        return (
            "Skin rashes can have many causes including irritation, infection, and allergies. "
            "Keep the area clean, avoid scratching, and notice any spreading, fever, or pain. "
            "You may upload an image for analysis — it may provide supportive insight, "
            "but it does not replace clinical evaluation."
        )



    # Pain
    if "pain" in msg or "ache" in msg or "hurts" in msg:
        return (
            "Pain varies in cause and severity. Gentle rest, hydration, and avoiding strain can help. "
            "Seek medical review if pain is severe, persistent, worsening, or linked to injury, fever, "
            "numbness, or weakness. I cannot diagnose conditions, but I can help guide next steps."
        )

    # Medication or treatment questions
    if "medicine" in msg or "medication" in msg or "take" in msg:
        return (
            "I cannot recommend or dose medications. Medication decisions depend on your history, "
            "allergies, and other conditions. A pharmacist or clinician is best placed to advise you. "
            "I can help discuss symptoms if that would be useful."
        )

    # General fallback
    return (
        "I can provide general health guidance and help you think about symptoms, "
        "but I cannot diagnose conditions. You can also try:\n"
        "- uploading an image for visual analysis,\n"
        "- recording your voice for vocal pattern analysis.\n"
        "Let me know what symptoms you are experiencing."
    )
