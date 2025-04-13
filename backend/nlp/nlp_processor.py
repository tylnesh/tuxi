# nlp/nlp_processor.py
import requests
import json
import re
from backend.config_manager import ConfigManager

def clean_text(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s([,.!?])', r'\1', text)
    # Ensure space after punctuation if missing
    text = re.sub(r'([,.!?])(?=[^\s])', r'\1 ', text)
    # Optionally, clean markdown formatting (if needed)
    text = re.sub(r'\s+(\*\*|\*)', r'\1', text)
    return text.strip()

class NLPProcessor:
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config  # Load the configuration
        self.model = self.config.get("nlp", {}).get("model", "gemma3:27b")
        self.api_host = self.config.get("api", {}).get("host", "localhost")
        self.api_port = self.config.get("api", {}).get("port", 11434)
  
    def query_stream(self, prompt: str, stream_callback) -> str:
        full_response = ""
        url = f"http://{self.api_host}:{self.api_port}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt
        }
        headers = {"Content-Type": "application/json"}
        try:
            with requests.post(url, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("response", "")
                        except Exception as parse_error:
                            print("Error parsing JSON chunk:", parse_error)
                            content = line
                        # Clean the chunk
                        content_clean = clean_text(content)
                        if content_clean:
                            stream_callback(content_clean)
                            # Optionally, add a space between chunks if needed.
                            if full_response and not full_response[-1].isspace():
                                full_response += " "
                            full_response += content_clean
                return full_response
        except Exception as e:
            print("Error calling Ollama REST API:", e)
            return ""


    def query_stop(self):
        url  = f"http://{self.api_host}:{self.api_port}/api/generate"
        payload = {
            "model": self.model,
            "keep_alive": -1
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            requests.post(url, json=payload, headers=headers)
        except Exception as e:
            print("Error calling Ollama REST API:", e)
            return ""
