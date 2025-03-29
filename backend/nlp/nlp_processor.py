# nlp/nlp_processor.py
import subprocess
import yaml
import os

# Global variables to hold the current subprocess and stop flag
current_nlp_proc = None
stop_streaming = False

class NLPProcessor:
    def __init__(self, config_file="config.yaml"):
        self.config = self.load_config(config_file)
        self.model = self.config.get("nlp", {}).get("model", "gemma3:27b")

    def load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file {path} not found.")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def query_stream(self, prompt: str, stream_callback) -> str:
        global current_nlp_proc, stop_streaming
        stop_streaming = False  # Reset the flag at the start
        full_response = ""
        command = ["stdbuf", "-oL", "ollama", "run", self.model, prompt]
        try:
            current_nlp_proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            # Read output line-by-line; check the stop flag on each iteration.
            for line in iter(current_nlp_proc.stdout.readline, ''):
                if stop_streaming:
                    break
                if line:
                    chunk = line.strip()
                    stream_callback(chunk)
                    full_response += chunk
            # After breaking, terminate if not already done.
            if current_nlp_proc.poll() is None:
                current_nlp_proc.terminate()
            current_nlp_proc = None
            return full_response
        except Exception as e:
            print("Error calling Ollama:", e)
            current_nlp_proc = None
            return ""
