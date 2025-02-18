import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Set the cache directory where model files will be stored locally
cache_dir = "./model_cache"
datafile = "support_cases.csv"
valfile = "validation_cases.csv"  # Assuming you have a separate validation dataset

# Ensure you're running the code in offline mode
os.environ["TRANSFORMERS_CACHE"] = cache_dir  # Set the cache directory path

# Step 1: Load the pre-trained model and tokenizer locally
model_name = 'gpt2'  # This can be any model you choose
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Set padding token to eos_token (End-Of-Sequence token for GPT-2)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load your dataset
train_data = pd.read_csv(datafile)
val_data = pd.read_csv(valfile)  # Load validation dataset
case_descriptions = train_data['case_description'].tolist()
resolutions = train_data['resolution'].tolist()

val_case_descriptions = val_data['case_description'].tolist()
val_resolutions = val_data['resolution'].tolist()

# Step 3: Prepare the dataset for fine-tuning
train_formatted_data = {
    "case_description": case_descriptions,
    "resolution": resolutions
}
val_formatted_data = {
    "case_description": val_case_descriptions,
    "resolution": val_resolutions
}

# Step 4: Create Datasets from the formatted data
train_dataset = Dataset.from_dict(train_formatted_data)
val_dataset = Dataset.from_dict(val_formatted_data)

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    # Tokenizing input descriptions, but also ensuring labels are shifted by one token
    encodings = tokenizer(examples['case_description'], padding="max_length", truncation=True, max_length=512)
    # Shift input_ids for causal language modeling
    encodings['labels'] = encodings['input_ids'].copy()
    encodings['labels'] = [x[1:] + [tokenizer.pad_token_id] for x in encodings['labels']]  # Shift labels by 1
    return encodings

train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
val_tokenized_datasets = val_dataset.map(tokenize_function, batched=True)

# Step 6: Fine-tune the model with the dataset
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,  # Log training every 100 steps
#    report_to="tensorboard",  # If you want to use TensorBoard for monitoring
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=val_tokenized_datasets,  # Pass the validation dataset
)

# Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model (Optional, after training)
model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

# Step 8: Function to generate a resolution for a new case after training
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def generate_resolution(case_description):
    # Ensure a clear prompt structure for the model
    input_text = f"Case Description: {case_description}\nResolution:"
    
    # Generate resolution using the trained model
    result = generator(input_text, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True, top_k=50)
    return result[0]['generated_text']


# Example of generating a resolution for a new case after training
new_case = "The application crashes while resetting the password."
predicted_resolution = generate_resolution(new_case)

print(f"\nPredicted Resolution:\n {predicted_resolution}")
