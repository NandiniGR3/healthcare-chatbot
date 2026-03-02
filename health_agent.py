import requests
import re

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
- Do NOT suggest drug names
- Always advise consulting a qualified doctor
- Keep responses clear and simple
"""

# ===============================
# AGENT MEMORY
# ===============================
chat_memory = []

# ===============================
# TOOL: LLM CALL
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
# AGENTIC REASONING LOOP
# ===============================
def health_agent(user_input):
    # ---- Normalize input ----
    clean_input = user_input.lower()
    clean_input = re.sub(r"[^a-zA-Z, ]", "", clean_input)

    # ---- Memory update ----
    chat_memory.append(f"User: {clean_input}")

    # ---- Reasoning context ----
    reasoning_prompt = f"""
Conversation Context:
{chr(10).join(chat_memory[-5:])}

Task:
Analyze the user's health concern and respond safely
without diagnosis or medication.
"""

    response = call_llm(reasoning_prompt)

    # ---- Memory update ----
    chat_memory.append(f"Assistant: {response}")

    return response

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("=" * 55)
    print("   Agentic Healthcare Assistant (LLM-only)")
    print("   Type 'exit' to quit")
    print("=" * 55)

    while True:
        user = input("\nYou: ")

        if user.lower() in ["exit", "bye"]:
            print("\nAgent: Stay healthy! ")
            break

        reply = health_agent(user)
        print("\nAgent:\n", reply)

if __name__ == "__main__":
    main()
