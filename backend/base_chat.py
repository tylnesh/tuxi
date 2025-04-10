from backend.nlp.nlp_processor import NLPProcessor
from backend.nlp.intent_parser import IntentParser
from backend.agents.calendar.calendar_agent import CalendarAgent


class BaseChat:
    def __init__(self, config_file="./backend/config/config.yaml"):
        self.config_file = config_file
        self.nlp_processor = NLPProcessor(config_file=config_file)
        self.intent_parser = IntentParser(config_file=config_file)

    def process_prompt(self, command: str):
        print(f"\n[Processing]: {command}")
        print("[Available Intents]:", self.intent_parser.get_all_intents())
        intent = self.intent_parser.parse_intent(command)
        print("\n[Detected Intent]:", intent)

        self.call_agent(intent, command)

    
    def call_agent(self, intent: str, prompt: str):
        """
        Call the agent with the given intent and prompt.
        """
        match intent:
            case "calendar_event":
                # Create an instance of CalendarAgent
                calendar_agent = CalendarAgent()
                calendar_agent.create_event_from_prompt(prompt)
            case _:
                print(f"[Warning] Unimplemented intent: {intent}, switching to general LLM.")
                self.general_llm(prompt)
                
    
    def general_llm(self, prompt: str):
        """
        Call the general LLM with the given prompt.
        """
        chunks = []

        def stream_callback(chunk: str):
            print(chunk, end=" ", flush=True)
            chunks.append(chunk)

        try:
            self.nlp_processor.query_stream(command, stream_callback)

            response = " ".join(chunks).strip()
            if not response:
                print("\n[Error] No response received from the model.")
                return

            print(f"\n\n[Final LLM Response]: {response}")
            print("-" * 40)
        except Exception as e:
            print(f"\n[Error]: {e}")
