import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from datasets import Dataset

# Set the cache directory where model files will be stored locally
cache_dir = "./model_cache"
datafile="support_cases.csv"

# Step 1: Download the model and tokenizer locally (Ensure this is done once while online)
# You can use the Hugging Face CLI to download models before running the script offline.
# Run this only when you have internet access:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

# Ensure you're running the code in offline mode
os.environ["TRANSFORMERS_CACHE"] = cache_dir  # Set the cache directory path

# Step 2: Load the pre-trained model and tokenizer locally
model_name = 'gpt2'  # This can be any model you choose
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Step 3: Initialize the pipeline for offline text generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

'''
# Step 4: Prepare your dataset (Case descriptions and resolutions)
data = [
    {"case_description": "Case 1: The defendant is accused of fraud.", "resolution": "Defendant found guilty of fraud."},
    {"case_description": "Case 2: A contract dispute over payment terms.", "resolution": "The court ruled in favor of the plaintiff, ordering payment."},
    {"case_description": "Case 3: A patent infringement case.", "resolution": "The defendant is required to pay damages to the plaintiff."},
    # Add more cases and resolutions here.
]
'''

# Step 4: Prepare your dataset (Case descriptions and resolutions)
data = pd.read_csv(datafile)
case_descriptions = data['case_description'].tolist()
resolutions = data['resolution'].tolist()
# Step 5: Convert the list of dictionaries into a format suitable for `Dataset.from_dict()`
'''
# Extract lists of case descriptions and resolutions
case_descriptions = [item["case_description"] for item in data]
print("case descriptio",case_descriptions)
resolutions = [item["resolution"] for item in data]
'''
# Create a dictionary where the keys are column names and the values are lists
formatted_data = {
    "case_description": case_descriptions,
    "resolution": resolutions
}

# Step 6: Create a Dataset from the formatted data
dataset = Dataset.from_dict(formatted_data)

# Step 7: Function to generate a resolution for a new case
def generate_resolution(case_description):
    # Format the input for text generation
    input_text = case_description + " Resolution:"
    
    # Generate resolution offline
    result = generator(input_text, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Example of generating a resolution for a new case
#new_case = "The user is unable to log in due to insufficient permissions."
new_case = "The application crashes while reset the password."
predicted_resolution = generate_resolution(new_case)
print(f"\nPredicted Resolution:\n {predicted_resolution}")
