import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Load and Preprocess Support Cases Data
def load_data(file_path):
    """Load and preprocess support cases data."""
    data = pd.read_csv(file_path)
    data = data[['case_description', 'resolution']]  # Ensure columns are case_description and resolution
    data['input_text'] = "Case: " + data['case_description'] + "\nResolution: "
    return data

# Step 2: Prepare Dataset for Fine-Tuning
def prepare_dataset(data):
    """Convert pandas DataFrame to Hugging Face Dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = Dataset.from_pandas(data[['input_text', 'resolution']])
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'resolution'])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset, tokenizer

# Step 3: Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained('gpt2')))

# Step 4: Fine-Tuning Function
def fine_tune_gpt2(tokenized_dataset, tokenizer):
    """Fine-tune the GPT-2 model."""
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
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

    # Adjust input length to avoid index out of range error
    max_input_length = model.config.n_positions - 1
    if inputs.size(1) > max_input_length:
        inputs = inputs[:, :max_input_length]

    outputs = model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1, 
        temperature=0.7, 
        pad_token_id=tokenizer.eos_token_id
    )

    resolution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resolution

# Main Execution
if __name__ == "__main__":
    # Load data
    file_path = "AIModel_supportSpec\servicenow_data.csv"  # Replace with your CSV file path
    data = load_data(file_path)

    # Prepare dataset
    tokenized_dataset, tokenizer = prepare_dataset(data)

    # Fine-tune GPT-2
    fine_tune_gpt2(tokenized_dataset, tokenizer)

    # Generate a resolution
    new_case = "User unable to access their account due to password reset issues."
    resolution = generate_resolution(new_case)
    print("Generated Resolution:\n", resolution)
