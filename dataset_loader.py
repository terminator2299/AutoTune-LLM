from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

# Configurations
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to your model
DATASET_NAME = "tatsu-lab/alpaca"  # Change to your dataset
SAVE_PATH = "data/processed_dataset.json"

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Load Dataset
print(f"Loading dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME)

# Load Tokenizer
print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocessing function
def preprocess_function(example):
    text = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
    return {"input_text": text, "tokenized": tokenizer(text, truncation=True, padding="max_length", max_length=512)["input_ids"]}

# Apply preprocessing
print("Processing dataset...")
processed_data = dataset["train"].map(preprocess_function)

# Save processed dataset
with open(SAVE_PATH, "w") as f:
    json.dump(processed_data["train"], f, indent=4)

print(f"Processed dataset saved at: {SAVE_PATH}")