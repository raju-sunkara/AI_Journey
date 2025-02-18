from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./t5-summarization-model")
tokenizer = T5Tokenizer.from_pretrained("./t5-summarization-model")

# Generate a summary
input_text = "User cannot log in to the system."
input_ids = tokenizer(f"summarize: {input_text}", return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Summary:", summary)