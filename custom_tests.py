import re
import nltk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load the tokenizer and model from the folder
model_path = "distilbert_ai_text_detector"  # Replace with the correct path if needed

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Text cleaning function (same as used during training)
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


# Get input text from the user
input_text = """
PUT OUR TEXT HERE
"""

# Preprocess the input text
input_text_cleaned = clean_text(input_text)

# Tokenize and encode the input text using the tokenizer
inputs = tokenizer(input_text_cleaned, return_tensors='pt', truncation=True, padding=True, max_length=512)

# Move input tensors to the same device as the model (e.g., GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

# Display the result
if prediction == 1:
    print("The input text is likely AI-generated.")
else:
    print("The input text is likely human-written.")
