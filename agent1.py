import requests

#  AGENT CONFIGURATION 

MODEL = "qwen2.5:3b"  
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a Healthcare Assistant Agent.

RULES (STRICT):
- Provide general health information only
- Suggest lifestyle changes and home remedies
- DO NOT diagnose diseases
- DO NOT prescribe medicines or dosages
- DO NOT give medical certainty
- ALWAYS advise consulting a qualified doctor
- Be calm, empathetic, and clear
"""

DISCLAIMER = (
    "\n\nDisclaimer: This information is for educational purposes only. "
    "It is not a medical diagnosis. Please consult a qualified doctor."
)

#  MEMORY 
chat_memory = []

#  LLM CALL 

def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False #Generate the entire reply first, then send it back as one complete response
        }
    )
    return response.json()["message"]["content"]

#  HEALTHCARE AGENT 

def healthcare_agent(user_input):
    chat_memory.append(f"User: {user_input}")

    prompt = (
        "\n".join(chat_memory)
        + "\n\nRespond according to healthcare safety rules."
    )

    response = ask_llm(prompt) + DISCLAIMER
    chat_memory.append(f"Assistant: {response}")

    return response

# CHAT INTERFACE 

def main():
    print("=" * 60)
    print("Healthcare Assistant Agent")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAgent: Take care and stay healthy!")
            break

        reply = healthcare_agent(user_input)
        print("\nAgent:\n", reply)
main()
