from transformers import pipeline
from backend.core.config_manager import ConfigManager

class IntentParser:
    def __init__(self, config_file: str = "config/config.yaml"):
        
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        # Path to your locally fine-tuned model
        # If "intent_model" isn't specified in config, default to "intent_model" folder.
        # self.model_path = self.config.get("intent", {}).get("model", "intent_model")
        self.model_path = "backend/core/nlp/intent_parser/v2"

        # Any labels you still want to keep track of from the config
        self.candidate_intents = self.config.get("intent", {}).get("candidate_intents", [
            "weather_query",
            "reminder",
            "calendar_event",
            "web_search"
        ])

        # Initialize a standard text-classification pipeline using your local model
        # Make sure your model folder has config.json and model.safetensors/pytorch_model.bin,
        # and optionally tokenizer files if you want to load the tokenizer from the same folder.
        self.classifier = pipeline(
            "text-classification",
            model=self.model_path,
            tokenizer="distilbert-base-uncased"
        )

    def get_all_intents(self) -> list:
        """
        Returns a list of all candidate intents from config.
        """
        return self.candidate_intents

    def parse_intent(self, text: str) -> str:
        """
        Uses the text-classification pipeline to infer the intent from the text.
        Returns the top predicted intent label as a string.
        """
        # The pipeline output is typically a list of dicts like:
        # [{"label": "...", "score": 0.9876}, ...]
        results = self.classifier(text)
        if not results:
            return "unknown"

        # Pick the highest-scoring label
        top_result = results[0]
        predicted_label = top_result["label"]

        return predicted_label
