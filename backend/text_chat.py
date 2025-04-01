import threading
from config_manager import ConfigManager
from nlp.nlp_processor import NLPProcessor
from nlp.intent_parser import IntentParser

class TextChat:
    def __init__(self, config_file="./config/config.yaml"):
        # Initialize real NLP components
        self.nlp_processor = NLPProcessor(config_file=config_file)
        self.intent_parser = IntentParser()

    def process_prompt(self, command: str):
        """
        Processes the given command and forwards it to the NLP model.
        Streams the NLP response in real-time without additional prefixes,
        and outputs the final intent.
        """
        print(f"[TextChat] Processing command: {command}")  # Debugging log
        full_response_chunks = []

        # Adjust the callback to print tokens with a trailing space.
        def stream_callback(chunk: str):
            print(chunk, end=" ", flush=True)
            full_response_chunks.append(chunk)

        try:
            # Stream output will be printed token by token.
            self.nlp_processor.query_stream(command, stream_callback)
            # Join the streamed tokens with a space to improve formatting.
            final_response = " ".join(full_response_chunks).strip()
            if not final_response:
                print("\n[Error] No response received from the model.")
                return
            # Print a newline before the final output.
            print("\n\n[Final LLM Response]:", final_response)
            intent, details = self.intent_parser.parse_intent(final_response)
            print("[Detected Intent]:", intent)
            print("[Intent Details]:", details)
            print("-" * 40)
        except Exception as e:
            print(f"\n[Error] Failed to process the command: {e}")

    def start(self):
        """
        Starts the text chat interface.
        Prompts the user for input and forwards it to the NLP model.
        """
        print("💬 Text Chat Interface Started. Type your commands below.")
        try:
            while True:
                user_input = input("You: ")
                if user_input.strip().lower() == "exit":
                    print("\n🛑 Exiting Text Chat.")
                    break
                if not user_input.strip():
                    print("[Info] Empty input. Please type something.")
                    continue
                self.process_prompt(user_input)
        except KeyboardInterrupt:
            print("\n🛑 Text Chat Stopped.")

if __name__ == "__main__":
    text_chat = TextChat()
    text_chat.start()