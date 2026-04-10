def chatbot_reply(user_msg: str) -> str:
    if not user_msg:
        return (
            "Hello — I can help with general health guidance. "
            "Tell me your symptoms, or try the image or voice analysis tools. "
            "This system does not provide medical diagnoses."
        )

    msg = user_msg.lower()

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

    categories = {
        "respiratory": {
            "keywords": ["cough", "breath", "wheezing"],
            "response": (
                "You mentioned cough or breathing symptoms. These can come from infections, "
                "allergies, or irritation. Stay hydrated and monitor breathing. "
                "Seek medical advice if symptoms worsen or persist."
            )
        },
        "fever": {
            "keywords": ["fever", "temperature", "flu"],
            "response": (
                "Fever is often linked to infection. Rest and fluids are important. "
                "Seek medical advice if it lasts more than 3 days or becomes very high."
            )
        },
        "skin": {
            "keywords": ["rash", "skin", "spots"],
            "response": (
                "Skin symptoms like rashes may be due to irritation, allergy, or infection. "
                "Avoid scratching and monitor for spreading or pain."
            )
        },
        "pain": {
            "keywords": ["pain", "ache", "hurts"],
            "response": (
                "Pain can have many causes. Rest and avoiding strain may help. "
                "Seek care if it is severe, persistent, or worsening."
            )
        },
        "medication": {
            "keywords": ["medicine", "medication", "take"],
            "response": (
                "Medication advice depends on your medical history. "
                "A pharmacist or clinician is best placed to guide you safely."
            )
        }
    }

    # Highest priority
    if any(term in msg for term in urgent_signs):
        return (
            "Some symptoms you mentioned could indicate a medical emergency. "
            "Please seek urgent medical attention or contact emergency services immediately."
        )

    # Detecting matching categories
    matched = []

    for name, data in categories.items():
        if any(keyword in msg for keyword in data["keywords"]):
            matched.append(data["response"])


    if matched:
        return (
            "Based on what you described:\n\n"
            + "\n\n".join(matched)
            + "\n\nIf symptoms worsen or you're unsure, seek medical advice. "
              "I cannot diagnose conditions, but I can help guide you."
        )


    return (
        "I can provide general health guidance, but I cannot diagnose conditions.\n"
        "You can also try:\n"
        "- uploading an image\n"
        "- recording your voice\n\n"
        "Tell me more about your symptoms."
    )