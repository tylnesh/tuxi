from backend.base_chat import BaseChat


class TextChat(BaseChat):
    def start(self):
        print("\n💬 Text Chat Interface Started. Type your commands below.")
        try:
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    print("\n🛑 Exiting Text Chat.")
                    break
                if not user_input:
                    print("[Info] Empty input. Please type something.")
                    continue
                self.process_prompt(user_input)
        except KeyboardInterrupt:
            print("\n🛑 Text Chat Stopped.")


if __name__ == "__main__":
    TextChat().start()
