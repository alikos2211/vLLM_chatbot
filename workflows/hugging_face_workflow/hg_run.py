from hg_chatbot import GPT2Chatbot
import time



bot = GPT2Chatbot()
print("GPT-2 Chatbot (type 'quit' to exit, 'clear' to reset)\n")

while True:
    user_input = input("You: ").strip()
    start = time.perf_counter()

    if not user_input:
        continue
    if user_input.lower() == "quit":
        end = time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        break
    elif user_input.lower() == "clear":
        bot.clear_history()
        print("History cleared.\n")
        end = time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
    elif user_input.lower() == "history":
        if not bot.history:
            print("No history yet.\n")
            end = time.perf_counter()
            print(f"Elapsed: {end - start} seconds")
        else:
            print("\n--- Chat History ---")
            for i, turn in enumerate(bot.history, 1):
                print(f"{i}. You: {turn['user']}")
                print(f"   Bot: {turn['bot']}")
            print("--------------------\n")
            end = time.perf_counter()
            print(f"Elapsed: {end - start} seconds")
    else:
        reply = bot.chat(user_input)
        print(f"Bot: {reply}\n")
        end = time.perf_counter()
        print(f"Elapsed: {end - start} seconds")