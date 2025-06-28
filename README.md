# 📊 Sentiment Analysis on Food Security in Malaysia | NLP🌾🇲🇾

![Python](https://img.shields.io/badge/python-3.11-blue)

Welcome to my **Food Security Sentiment Analysis** project!

In this project, I analyze public sentiment around food security issues in Malaysia by leveraging **Natural Language Processing (NLP)** and **Deep Learning** techniques. My goal is to build an effective **sentiment classification model** and deploy it as an interactive **web application**.

Whether you’re a data enthusiast eager to uncover public opinions, or a developer ready to harness the power of deep learning and NLP, **this sentiment analysis project is your gateway.** 🚀✨



## 🚀 Project Overview

### 🎯 Objectives

1. Translate, clean, and prepare a multilingual dataset of food security-related posts.
3. Score sentiment and detect high-risk content.
4. Extract key discussion topics via topic modeling.
5. Perform geospatial & temporal risk analysis.
6. Train and fine-tune deep learning classifiers.
7. Visualize insights in an interactive Power BI dashboard.
8. Deploy the final model within a Dockerized web app.

### 📚 Dataset Description

The dataset consists of multilingual social media posts related to food security in Malaysia, containing approximately 34,000 records. Each post includes metadata such as timestamp and location when available.


### 🛠️ Project Steps
1. **Data Cleaning and Preparation 🧹** :
   - translate all text data into English using Google Translate API, since the dataset contains posts in multiple languages
   - apply standard text cleaning procedures including expanding contractions, removing URLs, mentions, emojis, special characters, and stopwords
   - perform lemmatization to normalize words and reduce dimensionality
   - score sentiment with VADER (very positive, positive, negative, very negative)
   - detect high-risk or vulnerable posts by setting sentiment-score thresholds
   - discover key themes via LDA topic modeling, evaluating with perplexity and coherence.
   - Encode sentiment classes into numerical labels.
   - Tokenize input using BERT tokenizer.
   - Apply padding (only for embedding-based models requiring fixed-length input).
   - Handling Class Imbalance
   - Split the dataset into training, validation, and test sets.
     
3. **Model Building 🤖** :
   - fine-tune pretrained transformers:
      - BERT (Bidirectional Encoder Representations from Transformers)
      - RoBERTa (A Robustly Optimized BERT Pretraining Approach)
      - DistilBERT (a smaller, faster, and lighter version of BERT)
   - Implement early stopping and learning rate scheduling for stable and efficient training
     
4. **Model Evaluation ✅** :
   - Evaluate using metrics: Accuracy, Precision, Recall, F1-score
   - Visualize results using Confusion Matrix and ROC-AUC Curve
   - Perform cross-validation or train/test split validation
   - fine-tuning hyperparameters to improve performance
   
4. **Visualization and Reporting** 📈  
   - build an interactive Power BI dashboard for exploring sentiment, risk, and topic trends.
     
5. **Deployment 🌐** :
   - Deploy the best-performing model as a web app using **Streamlit**
   - Provide easy-to-use input forms and display prediction results clearly.
   - Host on **Streamlit Cloud**



## 🛠️ Tech Stack

- **Python 3**
- **Libraries**:
  - `pandas`, `numpy` — (data handling)
  - `nltk`, `spaCy`, `transformers` — NLP and text processing
  - `gensim`, `scikit-learn`  — Topic modeling
  - `TensorFlow`, `PyTorch` — Deep learning modeling
  - `Matplotlib`, `Seaborn`,`wordcloud`, `PowerBI` — Visualization
- **Deployment**: Streamlit Cloud / local


## 📊 Demo

Try out the app on **Streamlit Cloud:** [Demo](https://food-security-sentiment-analysis.streamlit.app/)

[![App Screenshot](https://github.com/ibtihel-dhaouadi/food-security-sentiment-analysis/blob/main/capture%20app.png)](https://food-security-sentiment-analysis.streamlit.app/)  

## 📂 Project Structure
```bash
├── data/                 # dataset (CSV/JSON)
├── notebooks/            # Jupyter notebooks for EDA, preprocessing, modeling
├── models/               # Saved models
├── dashboard/            # Power BI dashboard
├── app/                  # Streamlit app
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

## 🌟 Motivation

Food security is a critical issue affecting millions worldwide, and public sentiment can reveal valuable insights about food availability, accessibility, and concerns within Malaysia. By analyzing social media and public posts, this project aims to assist policymakers, NGOs, and stakeholders in making data-driven decisions to improve food security strategies.


## 🕵️‍♂️ Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/ibtihel-dhaouadi/Food-Security-Sentiment/issues).



---

Thanks for taking the time to check out my project! 🙌

If this project sparks your interest, don’t hesitate to drop a ⭐ and reach out with your ideas or questions — I’m all ears! 👂🔥

You can also visit my 🧑‍💻 [GitHub](https://github.com/ibtihel-dhaouadi) profile or 🏆 [Kaggle](https://www.kaggle.com/dhaouadiibtihel98) profile for more projects.

---



