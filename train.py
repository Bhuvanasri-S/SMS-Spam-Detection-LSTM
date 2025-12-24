from preprocessing import load_and_preprocess, tokenize_and_pad
from model import build_lstm_model
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = load_and_preprocess("spam.csv")

X, tokenizer = tokenize_and_pad(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = build_lstm_model(vocab_size=5000, input_length=X.shape[1])

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save model & tokenizer
model.save("lstm_spam_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
