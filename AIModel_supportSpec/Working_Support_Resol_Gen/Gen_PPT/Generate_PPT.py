from pptx import Presentation # install python-pptx using pip..just added so that I dont forget the package name.
from pptx.util import Inches
#To be Done.
# update to include NICE template
# Create a PowerPoint presentation
prs = Presentation()

# Slide 1: Title Slide
slide_layout = prs.slide_layouts[0]  # Title Slide Layout
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]

title.text = "Introduction to Machine Learning, Neural Networks & AI"
subtitle.text = "Applying Transformers & GPT-2 to Support Case Resolution\n Raju Sunkara| 10-Feb-2025 | NICE"

# Slide 2: Introduction to ML
slide_layout = prs.slide_layouts[1]  # Title and Content Layout
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Introduction to Machine Learning (ML)"
content.text = "- ML enables computers to learn from data.\n- Types:\n  • Supervised Learning\n  • Unsupervised Learning\n  • Reinforcement Learning"

# Slide 3: Neural Networks
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "What are Neural Networks?"
content.text = "- Neural networks mimic human brain neurons.\n- Components:\n  • Input Layer\n  • Hidden Layers\n  • Output Layer"

# Slide 4: AI Overview
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Introduction to Artificial Intelligence (AI)"
content.text = "- AI enables machines to perform tasks requiring intelligence.\n- AI includes:\n  • ML, NLP, Computer Vision, Robotics, etc.\n- Used in chatbots, self-driving cars, support ticket resolution."

# Slide 5: Evolution of AI
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Evolution of AI – Traditional ML to Transformers"
content.text = "- Early ML models required feature engineering.\n- Deep learning improved but had sequential limitations.\n- Transformers revolutionized NLP with self-attention."

# Slide 6: Understanding Transformers
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Understanding Transformers"
content.text = "- Self-attention mechanism enables parallel processing.\n- Most well-known Transformer model: GPT (Generative Pre-trained Transformer)."

# Slide 7: Pretrained Transformers
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Pretrained Transformers (GPT-2, GPT-3, T5)"
content.text = "- GPT-2: Generates human-like text.\n- T5: Converts NLP tasks to text-to-text format.\n- Pretrained models help in Support Case Resolution."

# Slide 8: GPT-2 for Support Cases
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "GPT-2 for Support Case Resolution"
content.text = "- GPT-2 fine-tuned on past support cases.\n- Steps:\n  • Collect data (Issue → Resolution)\n  • Tokenize & fine-tune GPT-2\n  • Use model to generate resolutions."

# Slide 9: Code Walkthrough (GPT-2 Fine-Tuning)
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Code Walkthrough: GPT-2 Fine-Tuning"
content.text = "1. Load Pretrained Model\n2. Prepare Support Case Dataset\n3. Fine-tune GPT-2\n4. Generate Resolutions"

# Slide 10: Evaluation Metrics
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Evaluation Metrics"
content.text = "- Loss (eval_loss)\n- BLEU/ROUGE Scores for text similarity\n- Human validation for correctness"

# Slide 11: Challenges & Considerations
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Challenges & Considerations"
content.text = "- Data quality affects results.\n- Model hallucination risks.\n- High computational cost.\n- Solutions: Domain-specific fine-tuning & human validation."

# Slide 12: Applications & Future Scope
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Applications & Future Scope"
content.text = "- Applications: Chatbots, AI Assistants, Ticket Categorization.\n- Future: GPT-4, RAG (Retrieval-Augmented Generation), and improved accuracy."

# Slide 13: Conclusion & Q/A
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Conclusion & Q/A"
content.text = "- AI is transforming customer support.\n- GPT-2 & Transformers provide automated resolutions.\n- Fine-tuning on domain-specific data improves performance.\n\nQuestions?"

# Adding speaker notes to each slide
slide_notes = {
    1: "This presentation provides an introduction to Machine Learning (ML), Neural Networks, and Artificial Intelligence (AI). We will also cover Transformers and their application in solving real-world support issues.",
    2: "Supervised Learning: The model is trained on labeled data. Example: Email spam detection.\n\n"
       "Unsupervised Learning: The model identifies patterns in unlabeled data. Example: Customer segmentation.",
    3: "Neural networks mimic the human brain. The diagram shows input layers, hidden layers, and output layers, demonstrating how information flows through the network.",
    4: "Transformers revolutionized NLP by enabling parallel processing and capturing long-range dependencies. The architecture includes layers of self-attention and feed-forward networks.",
    5: "This diagram illustrates the training process for our GPT-2 based support case resolution model.",
    6: "The model training involves tokenization, dataset preparation, fine-tuning, and evaluation.",
    7: "A walkthrough of the Python program, covering dataset preparation, model training, and evaluation using the Trainer API.",
    8: "Pretrained Transformers like GPT-2 can be used for resolving customer support issues by training on past cases to generate resolutions automatically.",
    9: "Summary of key takeaways from the session: AI and ML are transforming support automation, and fine-tuned transformers improve resolution generation.",
}

# Adding notes to each slide
for slide_number, notes in slide_notes.items():
    slide = prs.slides[slide_number - 1]
    slide.notes_slide.notes_text_frame.text = notes
# Save the presentation
pptx_filename = "AIModel_supportSpec\Working_Support_Resol_Gen\Gen_PPT\Output\AI_ML_Transformers_Presentation.pptx"
prs.save(pptx_filename)

pptx_filename