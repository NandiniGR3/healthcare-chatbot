import json
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# =====================================================
# Load FAQ Dataset (WITH KEYWORDS)
# =====================================================
with open(
    r"C:/Somalingaiah/nandinir/nlp-3rdsem/assignment/faq_chatbot_with_keywords.json",
    "r",
    encoding="utf-8"
) as file:
    faq_data = json.load(file)

# =====================================================
# Prepare Training Data from Keywords
# =====================================================
training_sentences = [] #The dataset has keywords, not full sentences.
training_labels = []

for intent, data in faq_data.items():
    for kw in data["keywords"]:
        examples = [
            f"i have {kw}",
            f"suffering from {kw}",
            f"having {kw}",
            f"{kw} problem",
            kw
        ]
        for ex in examples:
            training_sentences.append(ex.lower())
            training_labels.append(intent)

df = pd.DataFrame({
    "text": training_sentences,
    "intent": training_labels
}) #ML training dataset

# =====================================================
# ML Model (TF-IDF + Naive Bayes)
# =====================================================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), #unigrams + bigrams
        analyzer="char_wb" #character-level features
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
# Helper: Phrase-based keyword matching score --> Used for rule-based matching with confidence.
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
def format_response(intent, responses, confidence):
    return f"""
==============================
Possible Condition: {intent.replace("_", " ").title()}
Confidence: {confidence:.2f}
==============================

Description:
{responses[0]}

Home Remedies / Precautions:
{responses[1]}

When to Consult a Doctor:
{responses[2]}
""" + DISCLAIMER


# =====================================================
# Main Chatbot Logic
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

        rule_matched = False

        # ---------- Phrase-based Rule Matching ----------
        for intent, data in faq_data.items():
            keywords = data["keywords"]
            score = phrase_match_score(part, keywords)

            if score >= 0.20:   # phrase-level threshold
                detected_conditions[intent] = max(
                    detected_conditions.get(intent, 0),
                    round(score, 2)
                )
                rule_matched = True
        # ---------- ML Fallback ----------
        if not rule_matched: #probability scores for all possible intents.
            probs = model.predict_proba([part])[0]
            max_prob = float(np.max(probs))
            intent = model.classes_[np.argmax(probs)]

            if max_prob >= 0.45:
                detected_conditions[intent] = max(
                    detected_conditions.get(intent, 0),
                    round(max_prob, 2)
                )

    if not detected_conditions:
        return (
            "\nI’m not completely sure about your condition.\n"
            "Please describe your symptoms more clearly.\n"
            + DISCLAIMER
        )

    final_response = ""
    for intent, confidence in detected_conditions.items():
        final_response += format_response(
            intent,
            faq_data[intent]["responses"],
            confidence
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
