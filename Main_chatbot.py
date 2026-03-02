import json

# Load dataset
with open(r"C:/Somalingaiah/nandinir/nlp-3rdsem/assignment/faq_chatbot_with_keywords.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

DISCLAIMER = (
    "\nDisclaimer: This information is for educational purposes only. "
    "It is not a medical diagnosis. Please consult a professional doctor.\n"
)

def format_response(intent, responses):
    title = intent.replace("_", " ").title()

    return f"""
==============================
{title}
==============================

Description:
{responses[0]}

Home Remedies / Precautions:
{responses[1]}

When to Consult a Doctor:
{responses[2]}
""" + DISCLAIMER


def rule_based_chatbot(user_input):
    user_input = user_input.lower()
    matched_intents = []

    for intent, data in faq_data.items():
        keywords = data["keywords"]
        responses = data["responses"]

        for keyword in keywords:
            if keyword in user_input:
                matched_intents.append((intent, responses))
                break   # avoid duplicate matches for same intent

    if not matched_intents:
        return (
            "I'm sorry, I couldn't understand your concern.\n"
            "Please describe your symptoms clearly."
            + DISCLAIMER
        )

    response = ""
    for intent, responses in matched_intents:
        response += format_response(intent, responses)

    return response #Combines them into one output


# ================= Chatbot Interface =================
print("====================================")
print("Welcome to Healthcare FAQ Chatbot")
print("====================================")
print("Hello! I’m here to help you with common health-related queries.")
print("Please tell me what you’re feeling.\n")

while True:
    user = input("You: ")

    if user.lower() in ["exit", "quit", "bye"]:
        print("\nBot: Take care! Wishing you good health")
        print("Remember to consult a doctor for proper medical advice.")
        break

    print("\nBot:", rule_based_chatbot(user))
