#
#Prepared by Raju on 1st Feb 2025, surprisingly on weekend.. what to do??
#
#
import os # Import the os module to set environment variables
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Set the cache directory for offline use
cache_dir = "./model_cache"

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the app
CORS(app)  # To allow all domains to access the API, less secure but for testing purpose hah hah.. added kater when browser was complaining... 

# Load pre-trained GPT-2 tokenizer and model from local cache
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)

# Initialize the text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Route to generate resolution based on case description
@app.route('/generate_resolution/', methods=['POST'])
def generate_resolution():
    # Get case description from the incoming JSON request
    data = request.get_json()

    # Ensure the 'case_description' field is present in the request
    if 'case_description' not in data:
        return jsonify({"error": "case_description is required"}), 400
    
    case_description = data['case_description']
    
    # Prepare the input text for the model
    input_text = case_description + " Resolution:"
    
    # Generate the resolution text
    result = generator(input_text, max_length=100, num_return_sequences=1)
    
    # Extract the generated text from the result
    generated_text = result[0]['generated_text']
    
    # Return the generated resolution as a JSON response
    return jsonify({
        "case_description": case_description,
        "generated_resolution": generated_text
    })

# Run the Flask server (for development)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
