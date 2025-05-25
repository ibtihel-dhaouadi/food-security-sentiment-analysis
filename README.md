# 📊 Sentiment Analysis on Food Security in Malaysia | NLP

Welcome to our **Food Security Sentiment Analysis** project!

This project is a collaborative effort between [Onj Hajri](https://github.com/onsrajhi) and [ibtihel Dhaouadi](https://github.com/ibtihel-dhaouadi), aims to analyze public sentiment regarding **food security** in Malaysia using **Natural Language Processing (NLP)** and **Deep Learning (DL)**. The goal is to build a robust sentiment classification model and deploy it as a web application for public interaction.

Whether you’re a data enthusiast eager to uncover public opinions, or a developer ready to harness the power of deep learning and NLP, **this sentiment analysis project is your gateway.** 🚀✨


## 🚀 Project Overview

### 🎯 Objectives

1. Clean and prepare a real-world dataset of food security-related reviews or opinions in Malaysia.
2. Apply advanced NLP models to classify sentiment as **Positive**, **Neutral**, or **Negative**.
3. Evaluate and validate model performance.
4. Deploy the best-performing sentiment classifier as a user-friendly web application.

### 🛠️ Project Steps
1. **Data Cleaning and Preparation 🧹** :
   - Check if data is labeled (apply annotation or labeling strategy if not) 
   - Text cleaning (Remove noise, punctuation, emojis, special characters, and stopwords)
   - Apply tokenization, padding, and prepare word embeddings (GloVe / Word2Vec) 
   - Handle class imbalance if present (e.g., SMOTE or resampling) 
   - Split into train/test/validation sets
     
3. **Model Building 🤖** :
   - **Traditional baselines:** Logistic Regression, Support Vector Machine (SVM) with TF-IDF 
   - **Deep Learning Models:** LSTM (Long Short-Term Memory), Bi-LSTM (Bidirectional LSTM), CNN, Transformers-based models like BERT, RoBERTa, DistilBERT (fine-tuning pretrained models)
     
4. **Model Evaluation ✅** :
   - Use metrics: Accuracy, Precision, Recall, F1-score
   - Confusion Matrix and ROC-AUC Curve 
   - Cross-validation or train/test split validation
   - fine-tuning hyperparameters for best performance.
     
5. **Deployment 🌐** :
   - Deploy the best-performing model as a web app using **Streamlit**
   - Provide easy-to-use input forms and display prediction results clearly.
   - Host on **Streamlit Cloud**



## 🛠️ Tech Stack

- **Python 3**
- **Libraries**:
  - `pandas`, `numpy` (data handling)
  - `nltk`, `spaCy`, `transformers` — NLP and text processing 
  - `TensorFlow`, `PyTorch` (for deep learning modeling)
  - `scikit-learn` (traditional ML models)
  - `Matplotlib`, `Seaborn` (visualization)
- **Deployment**: Streamlit Cloud / local


## 📊 Demo

🚧 **Coming Soon** — The web app link will be available after deployment.

## 📂 Project Structure
```bash
├── data/                 # dataset (CSV/JSON)
├── notebooks/            # Jupyter notebooks for EDA and model building
├── models/               # Saved models
├── app.py                # Streamlit app
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 📦 How to Run Locally (Installation)

```bash
# Clone the repository
git clone https://github.com/ibtihel-dhaouadi/Food-Security-Sentiment.git
cd Food-Security-Sentiment

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

```




## 📌 Future Work
- ...
- ...



## 🕵️‍♂️ Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/ibtihel-dhaouadi/laptop-price-prediction-tn/issues).




## 🤝 Collaboration
This project is developed collaboratively by:

- Onj Hajri ➔   🧑‍💻 [GitHub](https://github.com/onsrajhi)  | 🔗 [LinkedIn](https://www.linkedin.com/in/onsrj/) 
- Dhaouadi Ibtihel ➔  🧑‍💻 [GitHub](https://github.com/ibtihel-dhaouadi) | 🔗 [LinkedIn](https://www.linkedin.com/in/ibtihel-dhaouadi/) 

---

<p align="center"> Thanks for taking the time to check out our project! 🙌 </p>

<p align="center"> If this project sparks your interest, don’t hesitate to drop a ⭐ and reach out with your ideas or questions — we’re all ears! 👂🔥 </p>

---



