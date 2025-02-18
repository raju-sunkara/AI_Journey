#Todo: 
# Parametertize the code to handle below inputs
#   dataset and validataion dataset
#   model_name
#   output_dir
#   max_input_length
#   max_target_length
#   training_args
#   Different model and tokenizer 
#   Adding more logging.

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset from CSV
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    # Rename columns to match the expected format for summarization
    df = df.rename(columns={"case_description": "input_text", "resolution": "target_text"})
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

# Preprocess the dataset for T5
def preprocess_data(dataset, tokenizer, max_input_length=512, max_target_length=150):
    def preprocess_function(examples):
        inputs = [f"summarize: {text}" for text in examples["input_text"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        # Tokenize targets
        labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# Load the T5 model and tokenizer
def load_model_and_tokenizer(model_name="t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Fine-tune the model
def fine_tune_model(train_dataset, eval_dataset, tokenizer, model, output_dir="./t5-summarization-model"):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Enable mixed precision if you have a GPU
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Main function
def main():
    # Path to your CSV files
    train_csv_file = "support_cases.csv"
    validation_csv_file = "validation_cases.csv"

    # Load datasets
    train_dataset = load_dataset(train_csv_file)
    validation_dataset = load_dataset(validation_csv_file)

    # Load T5 model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name="t5-small")

    # Preprocess data
    tokenized_train_dataset = preprocess_data(train_dataset, tokenizer)
    tokenized_validation_dataset = preprocess_data(validation_dataset, tokenizer)

    # Fine-tune the model
    fine_tune_model(tokenized_train_dataset, tokenized_validation_dataset, tokenizer, model)

    print("Model fine-tuning complete! Model saved to './t5-summarization-model'.")

if __name__ == "__main__":
    main()