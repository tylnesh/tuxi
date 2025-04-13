import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from concurrent.futures import ThreadPoolExecutor

# --- Settings ---
INPUT_FILE = "./train.csv"
OUTPUT_FILE = "./train_augmented.csv"
TARGET_PER_CLASS = 500
RANDOM_SEED = 42
NUM_WORKERS = 4
SYNTHETIC_RATIO = 0.25

random.seed(RANDOM_SEED)

# --- Load dataset ---
df = pd.read_csv(INPUT_FILE)

# --- Paraphrasing model: BART ---
paraphrase_model_name = "eugenesiow/bart-paraphrase"
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paraphrase_model = paraphrase_model.to(device)

# --- Text generation model: Mistral ---
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
generation_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
generation_tokenizer.pad_token = generation_tokenizer.eos_token  # ✅ Fix for padding
generation_model = AutoModelForCausalLM.from_pretrained(mistral_model_id, torch_dtype=torch.float16, device_map="auto")

# --- Prompts for new examples ---
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
    "unknown": "Give an example of an unclear or vague voice command.",
}

# --- Augmentation functions ---

def synonym_style_transform(text):
    words = text.split()
    if len(words) < 4:
        return text
    random.shuffle(words)
    return " ".join(words)

def generate_paraphrases(text, num_return_sequences=1, max_length=64):
    input_ids = paraphrase_tokenizer(
        f"paraphrase: {text}", return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    outputs = paraphrase_model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=5,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return [paraphrase_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def generate_synthetic(label, count):
    prompt = generation_prompts.get(label, "Give an example of a voice command.")
    results = []

    for _ in range(count):
        input_text = f"[INST] {prompt} [/INST]"
        inputs = generation_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            output = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # ✅ Explicitly pass attention mask
                max_new_tokens=40,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=generation_tokenizer.eos_token_id
            )

        decoded = generation_tokenizer.decode(output[0], skip_special_tokens=True)
        result = decoded.replace(input_text, "").strip()
        results.append((result, label))

    return results

def augment_label_group(label_group):
    label, group = label_group
    existing_texts = group['text'].tolist()
    current_count = len(existing_texts)
    to_generate = TARGET_PER_CLASS - current_count
    if to_generate <= 0:
        return []

    num_synthetic = int(to_generate * SYNTHETIC_RATIO)
    num_augmented = to_generate - num_synthetic

    print(f"▶ Augmenting '{label}': {current_count} → {TARGET_PER_CLASS} ({num_augmented} aug + {num_synthetic} synth)")

    generated_texts = set()
    augmented_rows = []

    # Synthetic generation
    synthetic_rows = generate_synthetic(label, num_synthetic)
    augmented_rows.extend(synthetic_rows)
    generated_texts.update([text for text, _ in synthetic_rows])

    # Paraphrase/synonym augmentation
    while len(generated_texts) < to_generate:
        original = random.choice(existing_texts)
        mode = random.choice(["paraphrase", "synonym"])
        if mode == "paraphrase":
            try:
                new_texts = generate_paraphrases(original, num_return_sequences=1)
            except Exception as e:
                print(f"⚠️ Paraphrasing failed: {e}")
                new_texts = []
        else:
            new_texts = [synonym_style_transform(original)]

        for new_text in new_texts:
            if new_text not in existing_texts and new_text not in generated_texts:
                augmented_rows.append((new_text, label))
                generated_texts.add(new_text)
            if len(generated_texts) >= to_generate:
                break

    return augmented_rows

# --- Parallel augmentation ---
label_groups = list(df.groupby('label'))
augmented_rows = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = executor.map(augment_label_group, label_groups)
    for result in results:
        augmented_rows.extend(result)

# --- Save result ---
df_augmented = pd.DataFrame(augmented_rows, columns=["text", "label"])
df_full = pd.concat([df, df_augmented], ignore_index=True)
df_full.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Augmented dataset saved to '{OUTPUT_FILE}' with {len(df_full)} total rows.")
