import yaml
import json
import re
from transformers import pipeline
from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
from ics import Calendar, Event
import caldav

#TODO: Add an option to select which calendar to use and an interactive mode for the user to fill in the details
# when the LLM fails to parse the prompt correctly, or the prompt is too vague or incomplete.


try:
    import jsonschema
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False

class CalendarAgent:
    # Define a JSON schema for validation
    # (We require these four keys; start_time must be a string, etc.)
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_time": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "description": {"type": "string"}
        },
        "required": ["title", "start_time", "duration_minutes", "description"]
    }

    def __init__(self):
        self.__load_config__()
        
    def __load_config__(self, config_file: str ="./backend/config/nextcloud.yaml"):
        """Load the configuration from the file or create a default one."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        nc_config = config.get('nextcloud', {})
        self.nextcloud_url = nc_config.get("url")
        self.username = nc_config.get("username")
        self.api_key = nc_config.get("api_key")
        self.calendar_name = nc_config.get("calendar_name")
        
        # self.llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
        self.llm = pipeline(
            "text-generation",
            model="microsoft/phi-4",
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
        
    # todo: somehow make the llm use current date and time and get rid of additional garbage responses
    def parse_prompt(self, prompt: str):
        """
        Parse the prompt to extract event details.
        """
        # Compose the LLM messages with instructions
        messages = [
            {
                "role": "system",
                "content": (
                    "Today is " + datetime.now().strftime("%Y-%m-%d") + ". "
                    "Time is " + datetime.now().strftime("%H:%M") + ". "
                    "You are a calendar assistant. "
                    "You are an assistant that extracts event details from natural language prompts. "
                    "Your output must be a valid JSON object with the following keys: "
                    "'title' (string), 'start_time' (ISO 8601 datetime string), "
                    "'duration_minutes' (integer), and 'description' (string). "
                    "Do not include any extra text, notes or explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        
        response = self.llm(messages, max_new_tokens=128, do_sample=True)
        print()
 
        # Existing code that tries to access the generated_text
        generated_text = response[0]["generated_text"][-1]["content"]
        print("LLM Response:", generated_text)
        
        # --- NEW (Regex extraction) ---
        # If there's a code block fence, remove it
        if "```" in generated_text:
            generated_text = generated_text.strip().split("```")[1] 

        # Use regex to find the first { ... } block
        json_match = re.search(r"(\{.*\})", generated_text, re.DOTALL)
        if json_match:
            extracted_json = json_match.group(1)
        else:
            # If we can't find braces, fall back to the entire string
            extracted_json = generated_text
        
        try:
            details = json.loads(extracted_json)

            # --- NEW (Schema validation) ---
            if SCHEMA_AVAILABLE:
                jsonschema.validate(details, self.schema)
            
            # Convert start_time from ISO8601 to a datetime object.
            details["start_time"] = date_parse(details["start_time"])
            # Convert duration_minutes to a timedelta and rename the key to 'duration'.
            details["duration"] = timedelta(minutes=int(details["duration_minutes"]))
            del details["duration_minutes"]
            return details
        except Exception as e:
            print("Error parsing LLM output:", e)
            return {}
        
    def generate_ics_event(self, details):
        """
        Generates an ICS event string from event details.
        """
        calendar = Calendar()
        event = Event()
        event.name = details.get("title", "Untitled Event")
        event.begin = details.get("start_time", datetime.now())
        event.duration = details.get("duration", timedelta(hours=1))
        event.description = details.get("description", "")
        calendar.events.add(event)
        return str(calendar)

    def add_event_to_nextcloud(self, ics_content):
        """
        Connects to the Nextcloud CalDAV server and adds the event.
        Uses the API key (app password) for authentication.
        """
        client = caldav.DAVClient(url=self.nextcloud_url, username=self.username, password=self.api_key)
        principal = client.principal()
        calendars = principal.calendars()
        for cal in calendars:
            print(f"Calendar: {cal.name}")
        if not calendars:
            raise Exception("No calendars found on the Nextcloud account.")
        # Select the desired calendar by name.
        calendar = next((cal for cal in calendars if cal.name == self.calendar_name), None)
        if not calendar:
            raise Exception(f"Calendar '{self.calendar_name}' not found.")
        calendar.add_event(ics_content)
        print("Event added successfully!")

    def create_event_from_prompt(self, prompt):
        """
        Processes the natural language prompt, creates an event, and adds it to Nextcloud.
        """
        details = self.parse_prompt(prompt)
        if not details:
            return "Could not parse event details."
        ics_content = self.generate_ics_event(details)
        self.add_event_to_nextcloud(ics_content)
        return "Calendar event created successfully!"

if __name__ == '__main__':
    # Instantiate the agent. The __init__ method will load the config from "config/nextcloud.yaml".
    agent = CalendarAgent()
    
    # Example prompt
    prompt = "Schedule meeting with Alex tomorrow at 3PM for 1 hour about quarterly review."
    result = agent.create_event_from_prompt(prompt)
    print(result)
