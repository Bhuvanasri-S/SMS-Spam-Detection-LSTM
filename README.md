🏠 SMS Spam Detection using LSTM  

📌 Project Overview  
This project detects whether an SMS message is **Spam** or **Not Spam (Ham)** using LSTM neural networks and NLP techniques. It helps users quickly identify spam messages.  

💼 Business Problem  
Manually identifying spam messages is time-consuming and prone to errors. Spam can lead to fraud, phishing, or unwanted marketing.  

🎯 Objective  
Build an LSTM-based deep learning model to classify SMS messages as spam or ham with high accuracy.  

📊 Dataset  
Source: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
Total Records: 5,574 messages  
Features: `label` (ham/spam), `message` (text)  

🛠️ Approach  
1. **Data Preprocessing** – Cleaning text, removing stopwords, lemmatization  
2. **Text to Sequences** – Tokenization and padding sequences  
3. **Model Building** – LSTM neural network  
4. **Training & Evaluation** – Accuracy, Precision, Recall, F1-score  
5. **Deployment** – Streamlit web app for real-time predictions  

📈 Results  
- Model predicts spam vs ham messages effectively  
- Real-time SMS classification via web interface  

📌 Key Learnings  
- LSTM can capture sequence patterns in text  
- Proper preprocessing improves model accuracy  
- Streamlit simplifies web deployment  

🚀 How to Run the Project  
```bash
pip install -r requirements.txt
streamlit run spamapp.py
