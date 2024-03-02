from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Input text

#input_text = "Happy Birth day"


input_text = "very bad weather today"



#input_text = "BERT is an amazing model that revolutionized natural language processing!"

# Tokenize input text
tokens = tokenizer(input_text, return_tensors='pt')

# Perform inference
with torch.no_grad():
    outputs = model(**tokens)

# Get predicted class
predicted_class = torch.argmax(outputs.logits).item()

# Map predicted class to label
label_map = {0: 'Negative', 1: 'Positive'}  # Example mapping for binary classification
predicted_label = label_map[predicted_class]

# Print predicted label
print("Predicted label:", predicted_label)

