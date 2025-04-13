from datasets import load_dataset
import random
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch
import numpy as np
from evaluate import load

data_files = {
    "train": "./train.csv",
    "validation": "./validation.csv",
    "test": "./test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

print(dataset)

print(dataset["train"][random.randint(0, len(dataset["train"]) - 1)])# Example Output:
# {'text': 'Search for the best Italian restaurants near me', 'label': 'web_search'}

print(dataset["validation"][random.randint(0, len(dataset["validation"]) - 1)])
# Example Output:
# {'text': 'Look up the population of Brazil', 'label': 'web_search'}

print(dataset["test"][random.randint(0, len(dataset["test"]) - 1)])
# Example Output:
# {'text': 'Find me the latest soccer news', 'label': 'web_search'}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",  # or "longest"
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Collect all labels from the 'train' split
unique_labels = list(set(tokenized_dataset["train"]["label"]))

# Map labels to integers
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

tokenized_dataset = tokenized_dataset.map(encode_labels)



num_labels = len(unique_labels)  # total distinct intents

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(device)  # Move model to GPU/CPU

training_args = TrainingArguments(
    output_dir="./distilbert-intent-model_v2",
    eval_strategy="epoch",         # Evaluate each epoch
    save_strategy="epoch",             # Save each epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True,         # Load best model after training
    metric_for_best_model="accuracy",    # We'll define this metric below
    greater_is_better=True,
)

accuracy_metric = load("accuracy")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,         # For data collation
    compute_metrics=compute_metrics
)

trainer.train()

test_metrics = trainer.evaluate(tokenized_dataset["test"])
print("Test Metrics:", test_metrics)


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt")
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]  # Return the intent label using id2label

# Example usage
sample_texts = [
    "Please open Chrome",
    "Remind me to drink water in 1 hour",
    "What's the weather this weekend?",
    "Do some random nonsense"
]

for txt in sample_texts:
    predicted = predict_intent(txt)
    print(f"Text: {txt}\nPredicted Intent: {predicted}\n")
    
model.save_pretrained("./distilbert-intent-model")