import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Correct Model Name
reqAIModel = 'distilgpt2'

# Step 1: Load and Preprocess Support Cases Data
def load_data(file_path):
    """Load and preprocess support cases data."""
    data = pd.read_csv(file_path)
    required_columns = ['case_description', 'resolution']
    
    # Ensure required columns exist
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"'{column}' column is missing from the provided CSV file.")

    data = data[required_columns]
    data['input_text'] = "Case: " + data['case_description'] + "\nResolution: "
    return data

# Step 2: Prepare Dataset for Fine-Tuning
def prepare_dataset(data):
    """Convert pandas DataFrame to Hugging Face Dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained(reqAIModel)   
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

    def tokenize_function(examples):
        inputs = tokenizer(
            examples['input_text'], 
            truncation=True, 
            padding='max_length', 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        )
        labels = tokenizer(
            examples['resolution'], 
            truncation=True, 
            padding='max_length', 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

    dataset = Dataset.from_pandas(data[['input_text', 'resolution']])
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'resolution'])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset, tokenizer

# Step 3: Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained(reqAIModel)
model.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained(reqAIModel)))

# Step 4: Fine-Tuning Function
def fine_tune_gpt2(tokenized_dataset, tokenizer):
    """Fine-tune the GPT-2 model."""
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        prediction_loss_only=True,
        evaluation_strategy="no",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")

# Step 5: Generate Resolution for a New Case
def generate_resolution(input_case, model_path="./fine_tuned_gpt2"):
    """Generate resolution for a new support case."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    input_text = f"Case: {input_case}\nResolution: "
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Generate attention mask manually
    
    # Adjust input length to avoid index out of range error
    max_input_length = min(tokenizer.model_max_length, model.config.n_positions - 1)
    if inputs.size(1) > max_input_length:
        inputs = inputs[:, :max_input_length]
        attention_mask = attention_mask[:, :max_input_length]

    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,  # Pass attention mask explicitly
        max_length=150, 
        num_return_sequences=1, 
        temperature=0.7, 
        do_sample=True,  # Enable sampling for temperature to take effect
        pad_token_id=tokenizer.eos_token_id
    )

    resolution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resolution

# Main Execution
if __name__ == "__main__":
    # Load data
    file_path = "AIModel_supportSpec/support_cases.csv"  # Replace with your CSV file path
    try:
        data = load_data(file_path)
    except KeyError as e:
        print(f"Error: {e}")
        exit(1)

    # Prepare dataset
    tokenized_dataset, tokenizer = prepare_dataset(data)

    # Fine-tune GPT-2
    fine_tune_gpt2(tokenized_dataset, tokenizer)

    # Generate a resolution
    new_case = "The application crashes while submitting a form"
    resolution = generate_resolution(new_case)
    print("Generated Resolution:\n", resolution)
