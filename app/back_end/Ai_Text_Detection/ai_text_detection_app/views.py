import os
import re
import nltk
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rest_framework.response import Response
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view
from rest_framework import status
from django.http import JsonResponse
from nltk.corpus import stopwords
import easyocr

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('C:/Users/M-Tech/Desktop/github/AI-Generated vs. Human-Written Text/model/distilbert_ai_text_detector')
tokenizer = DistilBertTokenizer.from_pretrained('C:/Users/M-Tech/Desktop/github/AI-Generated vs. Human-Written Text/model/distilbert_ai_text_detector')
# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Preprocessing function to clean text (same as used during training)
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

# Function to extract text from image using OCR
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])  # Use 'en' for English
    result = reader.readtext(image_path)
    text = ' '.join([item[1] for item in result])
    print(text)
    return text

@api_view(['POST'])
def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        save_path = os.path.join('media', image.name)  # Define your path here
        path = default_storage.save(save_path, image)  # Save the image to the server

        # Respond with the path where the image is stored
        return JsonResponse({'image_path': path})

    return JsonResponse({'error': 'No image provided'}, status=400)

# View function to make prediction based on text input
@api_view(['POST'])
def predict_text(request):
    if request.method == 'POST':
        input_text = request.POST.get('text', None)
        if not input_text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        # Clean and preprocess the input text
        cleaned_text = clean_text(input_text)

        # Tokenize the cleaned text using DistilBERT tokenizer
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # Make prediction (1 = AI-generated, 0 = Human-written)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)  # Apply softmax to get probabilities

            # Get the prediction and the associated probability
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item() * 100  # Convert probability to percentage

        if prediction == 1:
            result = f"The input text is likely AI-generated with {confidence:.2f}% confidence."
        else:
            result = f"The input text is likely human-written with {confidence:.2f}% confidence."

        return JsonResponse({'prediction': result, 'confidence': f"{confidence:.2f}%"})

# View function to process the image path and make a prediction
@api_view(['POST'])
def predict_text_from_image(request):
    if request.method == 'POST':
        # Get the image path from the request
        image_path = request.POST.get('image_path', None)
        if not image_path:
            return JsonResponse({'error': 'No image path provided'}, status=400)

        # Ensure the path is valid and exists
        if not os.path.exists(image_path):
            return JsonResponse({'error': 'Image path is invalid or file does not exist'}, status=400)

        # Extract text from the image using the provided path
        extracted_text = extract_text_from_image(image_path)

        if not extracted_text:
            return JsonResponse({'error': 'No text extracted from image'}, status=400)

        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)

        # Tokenize the cleaned text using DistilBERT tokenizer
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # Make prediction (1 = AI-generated, 0 = Human-written)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)  # Apply softmax to get probabilities

            # Get the prediction and the associated probability
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item() * 100  # Convert probability to percentage

        if prediction == 1:
            result = f"The input text is likely AI-generated with {confidence:.2f}% confidence."
        else:
            result = f"The input text is likely human-written with {confidence:.2f}% confidence."

        return JsonResponse({'prediction': result, 'extracted_text': extracted_text, 'confidence': f"{confidence:.2f}%"})

    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)





















