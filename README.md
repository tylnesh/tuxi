# tuxi
An AI assistant for Linux desktop

You can launch chat by running either one of:

- python3 -m backend.voice_chat
- python3 -m backend.text_chat

in the root of the project.

If you want to train your own intent model, run
- python3 training.py
from the ./training/intent_parser folder

The resulting model (config.json and model.safetensors) then copy over to the ./backend/nlp/intent_model folder
