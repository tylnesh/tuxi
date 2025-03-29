from transformers import pipeline

class IntentParser:
    def __init__(self, candidate_intents=None):
        # Define your candidate intents; adjust as needed.
        if candidate_intents is None:
            candidate_intents = [
                "web_search", 
                "open_application", 
                "play_music", 
                "set_alarm",
                "unknown"
            ]
        self.candidate_intents = candidate_intents
        # Initialize the zero-shot classifier.
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def parse_intent(self, text) -> tuple:
        """
        Uses zero-shot classification to infer the intent from the text.
        Returns a tuple: (intent, details).
        """
        result = self.classifier(text, candidate_labels=self.candidate_intents)
        # The classifier returns a dict with 'labels' (sorted by score) and 'scores'.
        intent = result["labels"][0]  # Choose the highest scoring label.
        details = text  # You can also return the full classification result if needed.
        return intent, details
