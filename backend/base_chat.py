from backend.nlp.nlp_processor import NLPProcessor
from backend.nlp.intent_parser import IntentParser


class BaseChat:
    def __init__(self, config_file="./backend/config/config.yaml"):
        self.config_file = config_file
        self.nlp_processor = NLPProcessor(config_file=config_file)
        self.intent_parser = IntentParser(config_file=config_file)

    def process_prompt(self, command: str):
        print(f"\n[Processing]: {command}")
        print("[Available Intents]:", self.intent_parser.get_all_intents())

        chunks = []

        def stream_callback(chunk: str):
            print(chunk, end=" ", flush=True)
            chunks.append(chunk)

        try:
            intent = self.intent_parser.parse_intent(command)
            print("\n[Detected Intent]:", intent)
            self.nlp_processor.query_stream(command, stream_callback)

            response = " ".join(chunks).strip()
            if not response:
                print("\n[Error] No response received from the model.")
                return

            print(f"\n\n[Final LLM Response]: {response}")
            print("-" * 40)
        except Exception as e:
            print(f"\n[Error]: {e}")
