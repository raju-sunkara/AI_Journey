import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Load Data
def load_servicenow_data(file_path):
    """Load ServiceNow data and return as a Pandas DataFrame."""
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

# Preprocess Data
def preprocess_data(data):
    """Prepare the data for training by creating specific subsets for each task."""
    # Task 1: Ticket Categorization
    categorization_data = data[['description', 'category']].dropna()

    # Task 2: Duplicate Ticket Detection
    duplicate_data = data[['ticket_id', 'description', 'is_duplicate']].dropna()

    # Task 3: Resolution Recommendations
    resolution_data = data[['description', 'resolution']].dropna()

    return categorization_data, duplicate_data, resolution_data

# Tokenization
def tokenize_data(tokenizer, data, task):
    """Tokenize data based on the task."""
    if task == 'classification':
        return tokenizer(data['description'].tolist(), padding=True, truncation=True, max_length=256)
    elif task == 'similarity':
        return tokenizer(data['description'].tolist(), padding=True, truncation=True, max_length=256)
    elif task == 'generation':
        return tokenizer(data['description'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")

# Prepare Dataset
def prepare_dataset(data, labels, tokenizer, task):
    """Convert data and labels to a Hugging Face Dataset."""
    dataset = Dataset.from_dict({
        'text': data,
        'labels': labels
    })

    tokenized_dataset = dataset.map(lambda x: {
        **tokenizer(x['text'], padding=True, truncation=True, max_length=256),
        'labels': x['labels']
    }, batched=True)

    return tokenized_dataset

# Load Model
def load_model(task, model_name="distilgpt2"):
    """Load a pretrained model for the given task."""
    if task == 'classification':
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    elif task == 'similarity':
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif task == 'generation':
        return AutoModelForCausalLM.from_pretrained(model_name)

# Adjust Tokenizer Padding Token
def adjust_tokenizer(tokenizer):
    """Ensure the tokenizer has a padding token."""
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Training
def train_model(model, tokenizer, dataset, output_dir, task):
    """Train the model for the specified task."""
    adjust_tokenizer(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

# Flask App for REST APIs
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_api():
    task = request.json.get('task')
    file_path = request.json.get('file_path')

    if task not in ['classification', 'similarity', 'generation']:
        return jsonify({"error": "Invalid task specified."}), 400

    # Load and preprocess data
    data = load_servicenow_data(file_path)
    categorization_data, duplicate_data, resolution_data = preprocess_data(data)

    # Initialize tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adjust_tokenizer(tokenizer)

    if task == 'classification':
        print("Training for Ticket Categorization...")
        categorization_dataset = prepare_dataset(
            categorization_data['description'], categorization_data['category'], tokenizer, task='classification'
        )
        model = load_model('classification', model_name)
        train_model(model, tokenizer, categorization_dataset, output_dir="./classification_model", task='classification')
    elif task == 'similarity':
        print("Training for Duplicate Ticket Detection...")
        duplicate_dataset = prepare_dataset(
            duplicate_data['description'], duplicate_data['is_duplicate'], tokenizer, task='similarity'
        )
        model = load_model('similarity', model_name)
        train_model(model, tokenizer, duplicate_dataset, output_dir="./similarity_model", task='similarity')
    elif task == 'generation':
        print("Training for Resolution Recommendations...")
        resolution_dataset = prepare_dataset(
            resolution_data['description'], resolution_data['resolution'], tokenizer, task='generation'
        )
        model = load_model('generation', model_name)
        train_model(model, tokenizer, resolution_dataset, output_dir="./generation_model", task='generation')

    return jsonify({"message": f"Training for {task} completed successfully."}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    task = request.json.get('task')
    input_text = request.json.get('input_text')

    if task not in ['classification', 'similarity', 'generation']:
        return jsonify({"error": "Invalid task specified."}), 400

    model_name = f"./{task}_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adjust_tokenizer(tokenizer)

    if task == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == 'similarity':
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == 'generation':
        model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model(**inputs)

    if task == 'classification' or task == 'similarity':
        predictions = torch.argmax(outputs.logits, dim=-1).item()
        return jsonify({"prediction": predictions}), 200
    elif task == 'generation':
        generated_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
