# 🤖 AI vs Human-Written Text Detector 🧠

This project is a **Django web application** integrated with a **DistilBERT-based NLP model** to detect whether a given piece of text was written by a **human** or **AI**. It also includes OCR support to extract text from images and classify it accordingly.

---

## 🚀 Features

- 📝 **Text Classification**: Classify any input text as either human-written or AI-generated.
- 🖼️ **OCR Integration**: Upload an image, extract embedded text using EasyOCR, and analyze its origin.
- 🧠 **Trained Transformer Model**: Fine-tuned DistilBERT model using Hugging Face and PyTorch.
- 📊 **Confidence Score**: Displays how confident the model is in its prediction.
- 🗃️ **REST API**: Predict via HTTP requests with text or image paths.
- 🧹 **Smart Preprocessing**: Advanced cleaning pipeline with stopword removal and noise reduction.

---

## 🧱 Tech Stack

| Component       | Tech                            |
|----------------|----------------------------------|
| Backend         | Django + Django REST Framework  |
| ML Framework    | PyTorch + Hugging Face Transformers |
| NLP Model       | DistilBERT (fine-tuned)         |
| OCR             | EasyOCR                         |
| Dataset         | [`shahxeebhassan/human_vs_ai_sentences`](https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences) |

---

## 🧠 Model Training Pipeline

- Cleaned dataset: removed URLs, HTML, symbols, and stopwords.
- Tokenized with `DistilBertTokenizer` and padded to 512 tokens.
- Trained a `DistilBertForSequenceClassification` with 2 output labels.
- 3 epochs with `AdamW` optimizer and learning rate scheduler.
- Achieved **accuracy >90%** on evaluation set.

## Training Snippet

```python
response = model(**batch)
loss = response.loss
loss.backward()
optimizer.step()
```


## 🧠 Prediction Logic

    Input text (or extracted image text) is:

        Cleaned using regex and stopword filtering.

        Tokenized with DistilBERT tokenizer.

        Passed into the model for classification.

    Output is either:

        "AI-generated" 🧠

        "Human-written" 🧍

    Includes confidence score as a percentage.


## 🔌 API Endpoints
📄 Text Prediction

POST /api/predict_text/

Payload:

text=The future of AI lies in human augmentation.

Response:

{
  "prediction": "The input text is likely human-written with 92.34% confidence.",
  "confidence": "92.34%"
}

🖼️ Image Prediction

POST /api/predict_image_text/

Step 1: Upload Image

POST /api/upload_image/
image = file upload

Step 2: Classify Image Text

POST /api/predict_text_from_image/

image_path=media/your_uploaded_image.png

Response:

{
  "prediction": "The input text is likely AI-generated with 88.90% confidence.",
  "extracted_text": "Artificial intelligence has revolutionized...",
  "confidence": "88.90%"
}

## 🛠️ Setup Instructions
1. 🧪 Install Dependencies

pip install django djangorestframework transformers torch easyocr nltk

2. 🔤 Download NLTK Resources

import nltk
nltk.download('stopwords')

3. 🛠️ Run Django App

python manage.py runserver

Ensure you update model loading paths in views.py if necessary.
## 🧪 Testing Locally

You can test the API with curl or Postman:

curl -X POST -F "text=Is this human-written?" http://localhost:8000/api/predict_text/
    
