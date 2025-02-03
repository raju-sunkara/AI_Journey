
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load distilgpt2 model
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
# Function to generate text
def distilgpt2_generate_text(text):
    input = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(
    fn=distilgpt2_generate_text,
    inputs="text",  # Simplified to string format for Textbox input
    outputs="text",  # Simplified to string format for text output
    title="DistilGPT-2 Text Generation",
    description="Generate text using the DistilGPT-2 model. Type a prompt and see how the model continues the text."
)
iface.launch()