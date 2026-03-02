import json
import re
import requests

# ===============================
# CONFIG
# ===============================
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a Health Assistant AI.
Rules:
- Provide general health information only
- Do NOT diagnose diseases
- Do NOT prescribe medicines
- Always advise consulting a doctor
"""

# ===============================
# LOAD KNOWLEDGE BASE
# ===============================
with open("C:\\Somalingaiah\\nandinir\\nlp-3rdsem\\assignment\\faq_chatbot_with_keywords.json", "r", encoding="utf-8") as f:
    FAQ_DB = json.load(f)

# ===============================
# AGENT MEMORY
# ===============================
chat_memory = []

# ===============================
# TOOL 1: KEYWORD CONFIDENCE
# ===============================
def keyword_confidence(text, keywords):
    matches = 0
    for kw in keywords:
        if kw.lower() in text:
            matches += 1
    return matches / len(keywords) if keywords else 0.0

# ===============================
# TOOL 2: JSON SEARCH
# ===============================
def json_tool(user_input):
    results = []

    for condition, data in FAQ_DB.items():
        conf = keyword_confidence(user_input, data["keywords"])
        if conf >= 0.20:
            results.append((condition, conf, data["responses"]))

    return results

# ===============================
# TOOL 3: OLLAMA LLM
# ===============================
def call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# ===============================
# AGENT REASONING LOOP
# ===============================
def health_agent(user_input):
    chat_memory.append(f"User: {user_input}")
    user_input = user_input.lower()
    user_input = re.sub(r"[^a-zA-Z, ]", "", user_input)

    # ---- Perception ----
    symptoms = re.split(r",|and|also", user_input)
    symptoms = [s.strip() for s in symptoms if s.strip()]

    agent_observation = []

    # ---- Tool usage: JSON ----
    for symptom in symptoms:
        matches = json_tool(symptom)
        for m in matches:
            agent_observation.append(m)

    # ---- Reasoning ----
    if agent_observation:
        response = ""
        for condition, conf, answers in agent_observation:
            response += f"""
==============================
Possible Condition: {condition.replace("_"," ").title()}
Confidence: {conf:.2f}
==============================

Description:
{answers[0]}

Home Remedies / Precautions:
{answers[1]}

When to Consult a Doctor:
{answers[2]}

Disclaimer: This information is for educational purposes only.
Please consult a professional doctor.

"""
    else:
        # ---- LLM fallback ----
        context = "\n".join(chat_memory[-5:])
        response = call_llm(context)

    # ---- Memory Update ----
    chat_memory.append(f"Assistant: {response}")

    return response

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("="*55)
    print("   Agentic Healthcare Assistant ")
    print("   Type 'exit' to quit")
    print("="*55)

    while True:
        user = input("\nYou: ")
        if user.lower() == "exit":
            print("\nAgent: Stay healthy!")
            break

        reply = health_agent(user)
        print("\nAgent:\n", reply)

if __name__ == "__main__":
    main()
