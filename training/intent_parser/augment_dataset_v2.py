import os
import pandas as pd
import random
import torch
import time
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from multiprocessing import Process, Queue, cpu_count

# --- Settings ---
INPUT_FILE = "./train.csv"
OUTPUT_FILE = "./train_augmented.csv"
CHECKPOINT_DIR = "./checkpoints"
TARGET_PER_CLASS = 500
RANDOM_SEED = 42
SYNTHETIC_RATIO = 0.25
PARAPHRASE_BATCH_SIZE = 16

torch.set_num_threads(4)
random.seed(RANDOM_SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Logging helper ---
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# --- Load dataset ---
df = pd.read_csv(INPUT_FILE)

# --- Load generation model ---
log("Loading Zephyr model for synthetic generation...")
generation_model_id = "HuggingFaceH4/zephyr-7b-beta"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_id, use_fast=False)
generation_tokenizer.pad_token = generation_tokenizer.eos_token
generation_model = AutoModelForCausalLM.from_pretrained(
    generation_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Prompts ---
generation_prompts = {
    "web_search": "Give an example of a web search query someone might type.",
    "open_application": "Give an example of a voice command to open an application.",
    "play_music": "Give an example of a voice command to play music.",
    "set_alarm": "Give an example of a voice command to set an alarm.",
    "send_message": "Give an example of a voice command to send a message.",
    "calendar_event": "Give an example of a voice command to create a calendar event.",
    "weather_query": "Give an example of a query asking about the weather.",
    "news_update": "Give an example of a query asking for news updates.",
    "reminder": "Give an example of a voice command to set a reminder.",
    "translation": "Give an example of a voice command to translate a phrase.",
    "unit_conversion": "Give an example of a command for converting units.",
    "calculator": "Give an example of a command for doing a math calculation.",
    "joke": "Give an example of a voice command asking for a joke.",
    "fact": "Give an example of a voice command asking for a fact.",
    "unknown": "Give an example of an unclear or vague voice command."
}

# --- Shared paraphrasing process ---
def paraphrasing_worker(input_queue, output_queue):
    tokenizer = AutoTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
    model = AutoModelForSeq2SeqLM.from_pretrained("eugenesiow/bart-paraphrase").to("cpu")
    model.eval()

    while True:
        batch = input_queue.get()
        if batch is None:
            break

        paraphrased = []
        for text in batch:
            input_ids = tokenizer(
                f"paraphrase: {text}", return_tensors="pt", padding=True, truncation=True
            ).input_ids
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=64,
                    num_beams=5,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            paraphrased.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        output_queue.put(paraphrased)

input_queue = Queue()
output_queue = Queue()
paraphraser = Process(target=paraphrasing_worker, args=(input_queue, output_queue))
paraphraser.start()

# --- Utility functions ---
def generate_synthetic(label, count):
    prompt = generation_prompts.get(label, "One-line voice command only. Output just the command.")
    results = []
    for _ in range(count):
        inputs = generation_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            output = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=generation_tokenizer.eos_token_id
            )
        decoded = generation_tokenizer.decode(output[0], skip_special_tokens=True).strip()
        results.append((decoded, label))
    return results

def augment_label_group(label_group):
    label, group = label_group
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{label}.csv")
    if os.path.exists(checkpoint_path):
        log(f"✅ Skipping '{label}' — checkpoint exists.")
        return []

    existing = set(group['text'].tolist())
    to_generate = TARGET_PER_CLASS - len(existing)
    if to_generate <= 0:
        return []

    num_synth = int(to_generate * SYNTHETIC_RATIO)
    num_para = to_generate - num_synth
    augmented = []

    augmented += generate_synthetic(label, num_synth)

    to_paraphrase = random.choices(list(existing), k=num_para)
    batches = [to_paraphrase[i:i+PARAPHRASE_BATCH_SIZE] for i in range(0, len(to_paraphrase), PARAPHRASE_BATCH_SIZE)]

    for batch in batches:
        input_queue.put(batch)
        paraphrased = output_queue.get()
        for text in paraphrased:
            if text not in existing:
                augmented.append((text, label))

    pd.DataFrame(augmented, columns=["text", "label"]).to_csv(checkpoint_path, index=False)
    return augmented

# --- Main execution ---
log("🚀 Starting augmentation...")
all_augmented = []
for label_group in tqdm(df.groupby("label")):
    all_augmented.extend(augment_label_group(label_group))

input_queue.put(None)  # Tell paraphrasing process to stop
paraphraser.join()

log("🔄 Merging datasets...")
augmented_files = [pd.read_csv(os.path.join(CHECKPOINT_DIR, f)) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".csv")]
df_aug = pd.concat(augmented_files, ignore_index=True)
df_full = pd.concat([df, df_aug], ignore_index=True)
df_full.to_csv(OUTPUT_FILE, index=False)

log(f"✅ Final dataset saved to '{OUTPUT_FILE}' with {len(df_full)} rows.")
