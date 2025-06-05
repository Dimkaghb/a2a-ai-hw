# main.py

from workflows.interaction import A2AInteraction

def main():
    print("🤖 Привет! Я AI-агент. Спроси что-нибудь.")
    a2a = A2AInteraction()

    while True:
        question = input("\n🧠 Ты: ")
        if question.lower() in ["выход", "exit", "quit"]:
            break

        answer = a2a.ask(question)
        print(f"\n🗨️  Агент: {answer}")

if __name__ == "__main__":
    main()
