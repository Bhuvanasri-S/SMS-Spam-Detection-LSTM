import streamlit as st
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('stopwords', quiet=True)

nltk.download('wordnet', quiet=True)

# Load trained LSTM model and tokenizer
model = load_model("lstm_spam_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Text preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit page config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# UI
st.title("📩 SMS Spam Detection ")
st.write("Detect whether a message is **Spam** or **Not Spam**."
  
)

user_input = st.text_area(
    "Enter SMS message to check:",
    placeholder="Type your message here..."
)

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=100)

        probability = model.predict(padded)[0][0]

        st.write(f"Spam Probability: **{probability:.2f}**")

        if probability > 0.5:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM (HAM)")

st.markdown("---")
st.caption("Developed by Bhuvana | NLP & LSTM Project")
