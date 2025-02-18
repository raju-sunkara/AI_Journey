import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Set the cache directory where model files will be stored locally
cache_dir = "./model_cache_t5"
datafile = "support_cases.csv"  # Your dataset file

# Ensure you're running the code in offline mode
os.environ["TRANSFORMERS_CACHE"] = cache_dir  # Set the cache directory path

# Step 2: Load the pre-trained T5-small model and tokenizer locally
model_name = 't5-small'  # Using T5-small
model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Step 3: Initialize the pipeline for offline text generation
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Step 4: Load and prepare your dataset (Case descriptions and resolutions)
data = pd.read_csv(datafile)
case_descriptions = data['case_description'].tolist()
resolutions = data['resolution'].tolist()

# Step 5: Convert the list of dictionaries into a format suitable for Dataset.from_dict()
formatted_data = {
    "case_description": case_descriptions,
    "resolution": resolutions
}

# Step 6: Create a Dataset from the formatted data
dataset = Dataset.from_dict(formatted_data)

# Tokenize the dataset
def tokenize_function(examples):
    # Add "problem:" prefix for input text and "resolution:" for target text
    inputs = [f"problem: {desc}" for desc in examples['case_description']]
    targets = [f"resolution: {res}" for res in examples['resolution']]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
    
    # Ensure the labels are returned properly
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 7: Define the training arguments for T5 fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # You can use 'steps' or 'epoch' based on your needs
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Step 8: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    # Optionally add an eval_dataset if you have validation data
)

# Step 9: Train the model
trainer.train()

# Step 10: Function to generate a resolution for a new case
def generate_resolution(case_description):
    # Format the input for T5, it expects a prefix like "problem:"
    input_text = "problem: " + case_description
    
    # Generate resolution using the T5 model
    result = generator(input_text, max_length=150, num_return_sequences=1)
    return result[0]['generated_text']

# Example of generating a resolution for a new case
new_case = "The application crashes while resetting the password."
predicted_resolution = generate_resolution(new_case)
print(f"Predicted Resolution:\n {predicted_resolution}")
