import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def load_and_preprocess(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(clean_text)

    return df

def tokenize_and_pad(texts, max_len=100):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    return padded, tokenizer
