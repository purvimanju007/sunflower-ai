#  Sunflower Disease Detection using AI

## Project Overview
This project is an AI-powered application that detects diseases in sunflower leaves using deep learning.  
Users can upload or capture images of leaves, and the system predicts the disease along with confidence, advice, and multilingual support.
##  Features
-  Image-based disease detection
-  Upload image or use camera
-  Multilingual output (English, Hindi, Kannada, Tamil)
-  Chatbot for crop-related queries
-  Voice assistant (speech-to-text + text-to-speech)
- Confidence score with progress bar
## Technologies Used
- Python
- PyTorch (ResNet101 CNN)
- Streamlit (Frontend)
- Google Translate API (googletrans)
- SpeechRecognition
- gTTS (Text-to-Speech)
- gdown (for model download)
## How it Works
1. User uploads or captures a leaf image
2. Image is preprocessed and passed to the trained CNN model
3. Model predicts the disease class
4. Prediction is displayed with confidence score
5. Advice is provided based on prediction
6. Output can be translated into selected language
7. Chatbot and voice assistant provide additional help
## How to Run Locally

### Install dependencies
```bash
pip install -r requirements.txt