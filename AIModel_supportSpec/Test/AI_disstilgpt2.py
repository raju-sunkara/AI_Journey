from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')


prompt = """How exactly do I reset my password in servicenow?"""
pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_length=1024)
result = pipe(prompt)
print(result[0]['generated_text'])