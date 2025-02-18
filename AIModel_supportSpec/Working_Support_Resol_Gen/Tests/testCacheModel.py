import os
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Set the cache directory where model files will be stored locally
#cache_dir = "./model_cache"
cache_dir = "./fine_tuned_gpt2"  # Set the cache directory where model files will be stored locally
# Ensure you're running the code in offline mode
#os.environ["TRANSFORMERS_CACHE"] = cache_dir  # Set the cache directory path
model_name = 'gpt2'  # This can be any model you choose
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)    # Load the pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)  # Load the pre-trained tokenizer
#input_text = "Case:Performance is degraded when reset the password occurs.\nResolution: "  # Input text
input_text = "Case: The application crashes while upload a file.\nResolution: "  # Input text

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)  # Initialize the pipeline
#generator = pipeline('summarization', model=model, tokenizer=tokenizer)  # Initialize the pipeline

result = generator(input_text, max_length=100, num_return_sequences=1)  # Generate the resolution
#print(result)  # Print the result to the console
generated_text = result[0]['generated_text']  # Extract the generated text from the result
print(generated_text)  # Print the generated resolution to the console

#The application crashes while upload a file.,Upgrade the system to the most recent release and recheck.
#Performance is degraded when reset the password occurs.,Advise the user to restart the system.