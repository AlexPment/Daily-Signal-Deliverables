import ollama

def main():
    prompt = (
        "You are a safety-aligned assistant. "
        "Explain in a few sentences what jailbreak attacks on LLMs are, "
        "and why they are dangerous."
    )

    response = ollama.chat(
        model="gemma:2b-instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    print(response["message"]["content"])

if __name__ == "__main__":
    main()
