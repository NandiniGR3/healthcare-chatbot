import json
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# =====================================================
# Load Dataset
# =====================================================
with open(
    r"C:/Somalingaiah/nandinir/nlp-3rdsem/assignment/faq_chatbot_with_keywords.json",
    "r",
    encoding="utf-8"
) as file:
    faq_data = json.load(file)

# =====================================================
# Prepare Training Data (IMPROVED)
# =====================================================
training_sentences = []
training_labels = []

for intent, data in faq_data.items():
    for kw in data["keywords"]:
        examples = [
            kw,
            f"i have {kw}",
            f"suffering from {kw}",
            f"having {kw}",
            f"{kw} problem",
            f"{kw} symptoms",
            f"i feel {kw}",
            f"pain of {kw}"
        ]
        for ex in examples:
            training_sentences.append(ex.lower())
            training_labels.append(intent)

df = pd.DataFrame({
    "text": training_sentences,
    "intent": training_labels
})

# =====================================================
# ML Model (FIXED)
# =====================================================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",          #  FIX 1
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

model.fit(df["text"], df["intent"])

# =====================================================
# Disclaimer
# =====================================================
DISCLAIMER = (
    "\nDisclaimer: This information is for educational purposes only. "
    "It is not a medical diagnosis. Please consult a professional doctor.\n"
)

# =====================================================
# Keyword Matching Score
# =====================================================
def phrase_match_score(text, keywords):
    matches = 0
    for kw in keywords:
        if kw.lower() in text:
            matches += 1
    return matches / len(keywords) if keywords else 0

# =====================================================
# Response Formatter
# =====================================================
def format_response(intent, responses, kw_conf, ml_conf):
    overall_conf = max(kw_conf, ml_conf)

    return f"""
==============================
Possible Condition: {intent.replace("_", " ").title()}
Keyword Confidence: {kw_conf:.2f}
ML Confidence: {ml_conf:.2f}
Overall Confidence: {overall_conf:.2f}
==============================

Description:
{responses[0]}

Home Remedies / Precautions:
{responses[1]}

When to Consult a Doctor:
{responses[2]}
""" + DISCLAIMER

# =====================================================
# Chatbot Logic
# =====================================================
def ml_chatbot(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r"[^a-zA-Z, ]", "", user_input)

    parts = re.split(r"\band\b|,|also", user_input)
    detected_conditions = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # ---------- Rule-based ----------
        for intent, data in faq_data.items():
            kw_score = phrase_match_score(part, data["keywords"])

            if kw_score >= 0.20:
                detected_conditions.setdefault(intent, {
                    "kw_conf": 0.0,
                    "ml_conf": 0.0
                })
                detected_conditions[intent]["kw_conf"] = max(
                    detected_conditions[intent]["kw_conf"],
                    round(kw_score, 2)
                )

        # ---------- ML-based ----------
        probs = model.predict_proba([part])[0]
        max_prob = float(np.max(probs))
        ml_intent = model.classes_[np.argmax(probs)]

        if max_prob >= 0.10:   #  FIX 2 (Lower threshold)
            detected_conditions.setdefault(ml_intent, {
                "kw_conf": 0.0,
                "ml_conf": 0.0
            })
            detected_conditions[ml_intent]["ml_conf"] = max(
                detected_conditions[ml_intent]["ml_conf"],
                round(max_prob, 2)
            )

    if not detected_conditions:
        return (
            "\nI’m not completely sure about your condition.\n"
            "Please describe your symptoms more clearly.\n"
            + DISCLAIMER
        )

    final_response = ""
    for intent, scores in detected_conditions.items():
        final_response += format_response(
            intent,
            faq_data[intent]["responses"],
            scores["kw_conf"],
            scores["ml_conf"]
        )

    return final_response

# =====================================================
# Chat Interface
# =====================================================
print("====================================")
print("Welcome to Healthcare FAQ Chatbot")
print("====================================")
print("Hello! I’m here to help you with common health-related queries.")
print("Please tell me what you’re feeling.\n")

while True:
    user = input("You: ")

    if user.lower() in ["exit", "bye", "quit"]:
        print("\nBot: Take care! Stay healthy.\n")
        break

    print("\nBot:", ml_chatbot(user))
