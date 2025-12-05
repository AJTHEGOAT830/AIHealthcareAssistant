def chatbot_reply(user_msg):
    msg = user_msg.lower()

    if "cough" in msg or "breath" in msg:
        return "Cough or breathing symptoms have many causes. I cannot diagnose, but staying hydrated and monitoring symptoms may help."

    if "rash" in msg or "skin" in msg:
        return "Skin rashes are common. You may upload an image for analysis."

    if "pain" in msg:
        return "Pain can vary in severity. I cannot diagnose but resting and hydrating may help."

    return "I can provide general health guidance but not a diagnosis. You can use the voice or image analysis tools for additional insight."
