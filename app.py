import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from googletrans import Translator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import tempfile
import gdown   # ✅ ADDED

# Page config
st.set_page_config(page_title="Sunflower AI", layout="centered")

# ---------------- MODEL DOWNLOAD ----------------
model_path = "sunflower_model.pth"

url = "https://drive.google.com/uc?export=download&id=1cBGHMGgTtYTngI0HKA4jxwHshyx5qzNs"

# Download only if not present
import os
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# ---------------- LOAD MODEL ----------------
model = models.resnet101(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

classes = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']

translator = Translator()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Title
st.markdown("<h1 style='text-align: center;'>🌻 Sunflower Disease Detection</h1>", unsafe_allow_html=True)

# Language
language = st.selectbox("Select Language", ["English", "Hindi", "Kannada", "Tamil"])

# ---------------- IMAGE INPUT ----------------
option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# ---------------- PREDICTION ----------------
if image:
    st.image(image, caption="Input Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    prediction_text = classes[pred.item()]

    # Translation
    if language != "English":
        translated = translator.translate(prediction_text, dest=language.lower()).text
        st.success(f"Prediction: {translated}")
    else:
        st.success(f"Prediction: {prediction_text}")

    st.info(f"Confidence: {confidence.item()*100:.2f}%")
    st.progress(int(confidence.item()*100))

    solutions = {
        "Downy mildew": "Use fungicide and avoid excess moisture.",
        "Fresh Leaf": "Healthy leaf. No action needed.",
        "Gray mold": "Remove infected parts and improve air circulation.",
        "Leaf scars": "Check for pest damage and apply treatment."
    }

    advice = solutions[prediction_text]

    if language != "English":
        advice = translator.translate(advice, dest=language.lower()).text

    st.warning(f"Advice: {advice}")

# ---------------- CHATBOT ----------------
st.markdown("---")
st.subheader("💬 Ask about your crop")

def chatbot_response(query):
    query = query.lower()

    if any(word in query for word in ["rust", "fungus"]):
        return "Rust is a fungal disease. Apply fungicide and reduce moisture."
    elif "blight" in query:
        return "Blight spreads fast. Remove infected leaves immediately."
    elif "mildew" in query:
        return "Mildew occurs in humid conditions. Improve air circulation."
    elif "healthy" in query:
        return "Your plant appears healthy. Maintain proper care."
    elif "water" in query:
        return "Avoid overwatering. Ensure proper drainage."
    elif "fertilizer" in query:
        return "Use balanced NPK fertilizer for sunflower growth."
    elif "soil" in query:
        return "Sunflower grows best in well-drained loamy soil."
    else:
        return "Please ask about plant disease, watering, or crop care."

user_input = st.text_input("Ask a question:")

if user_input:
    response = chatbot_response(user_input)

    if language != "English":
        response = translator.translate(response, dest=language.lower()).text

    st.write("Bot:", response)

# ---------------- VOICE ASSISTANT ----------------
st.markdown("---")
st.subheader("🎤 Voice Assistant")

audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

if audio:
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio['bytes'])
        temp_audio_path = temp_audio.name

    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            st.write("You said:", text)

            response = chatbot_response(text)

            if language != "English":
                response = translator.translate(response, dest=language.lower()).text

            st.write("Bot:", response)

            tts = gTTS(response)
            tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tts_file.name)

            st.audio(tts_file.name)

        except:
            st.error("Could not understand audio")