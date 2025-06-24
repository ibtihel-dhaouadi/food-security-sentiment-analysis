import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.base import BaseEstimator, TransformerMixin
import contractions
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random

# --- Text Preprocessor Class ---
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words("english"))

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def expand_contractions(self, text):
        return contractions.fix(text)

    def preprocess_text(self, text):
        text = self.expand_contractions(text)
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        text = re.sub(r"[^\w\s]", "", text)
        tokens = word_tokenize(text)
        clean_tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words_en and word.isalpha()
        ]
        return " ".join(clean_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, str):
            return self.preprocess_text(X)
        elif isinstance(X, list):
            return [self.preprocess_text(text) for text in X]
        elif hasattr(X, "apply"):
            return X.apply(self.preprocess_text)
        else:
            raise TypeError("Input must be string, list, or pandas Series")

# --- Load Model and Tokenizer ---
@st.cache_resource(show_spinner=False)
def load_model_tokenizer(model_dir="Models/DistilBert_sentiment_model"):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

# --- Load Pipeline ---
@st.cache_resource(show_spinner=False)
def load_text_pipeline(pipeline_path="Models/text_preprocessor_pipeline.joblib"):
    return joblib.load(pipeline_path)

# Sample tweets for sentiment page
sample_tweets = [
    "No food at home again. Feeling hopeless. 😞",
    "Thanks to the generous volunteers, our community now has access to fresh meals every day, which is a blessing beyond words! 🙏❤️",
    "The prices for basic groceries have doubled this month. Many families in my neighborhood are struggling to put food on the table, and it's heartbreaking to witness.",
    "Got my free lunch today, feeling grateful! 😊",
    "Skipped dinner again, no money left for food.",
    "Amazing support from local charities providing food packages to the vulnerable. It really makes a difference in people's lives. #ThankYou",
    "The drought has destroyed the crops, leaving farmers with no income and communities facing severe food shortages.",
    "Fresh fruits and veggies at the food bank today! 🥳",
    "Empty fridge, empty stomach.",
    "Community kitchens are opening up and feeding hundreds daily. It's inspiring to see people come together to fight hunger and support each other.",
    "My 3 kids haven't had a proper meal in 2 days... 💔 #malnutrition",
    "Thank you @NGOFeeds for the RM100 food voucher 🧡 #solidarity #HopeMatters",
    "🔥 45% of rural families live in food insecurity. Read more 👉 www.helpinghands.org/stats",
    "#HungerAlert 🚨 Schools in Sabah report 1 in 4 students skipping lunch due to poverty.",
    "Day 6 without meat. Surviving on plain rice & water... 😩 #B40Voices",
    "Another night with no dinner. This is my 3rd day in a row. 😢 #Hungry",
    "Food delivery from the NGO just arrived! Eggs, rice, veggies—thank you so much!! 🥚🍚🥦 #Lifesaver",
    "Still waiting for @GovMalaysia to take action. Food aid promised last month never came! 😠",
    "Received a food basket with essentials today—oil, milk, sugar. Can't express how thankful I am! 🧺✨",
    "Prices have gone up by 70%! How are we supposed to feed our kids? 😩 #Inflation #FoodInsecurity",
    "https://helpnow.org is doing a great job feeding people in rural areas. Please support them! 🌾 #SupportLocal",
    "Got enough to cook 3 meals today for the first time this week. I'm literally in tears. 🥹❤️",
    "Just visited the food center—long queues, frustrated faces, empty shelves... 😔",
    "Shoutout to @FeedMYPeople for delivering care packages every weekend! 🥰 #Heroes",

]
# --- EDA Page ---
def eda_page():

    st.markdown("<h1 style='text-align: center;'>📊 Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:17px; color:gray;'> Explore, analyze & understand public sentiment on food insecurity </p>", unsafe_allow_html=True)

    # Load preprocessed dataset once with caching
    @st.cache_data(show_spinner=True)
    def load_data():
        df = pd.read_csv("dataset/food_security_processed_data.csv", parse_dates=["Datetime"])
        return df

    df = load_data()
    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Show raw data (first 100 rows)"):
        st.dataframe(df.head(100))

    st.subheader("Descriptive Statistics")
    st.write(df.describe(include="all"))



   # Number of tweets over time
    st.subheader("Number of Tweets Over Time")
    tweets_per_day = df.groupby(df["Datetime"].dt.date).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    tweets_per_day.plot(ax=ax, color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Tweets per Day")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)



    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_order = ['very negative', 'negative', 'positive', 'very positive']
    sentiment_colors = {
        'very negative': '#D62728',  # red
        'negative': '#FF7F0E',       # orange
        'positive': '#2CA02C',       # green
        'very positive': '#1F7A1F'   # dark green
    }
    sentiment_counts = df["sentiment_label"].value_counts().reindex(sentiment_order).fillna(0)
    fig2, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", color=[sentiment_colors.get(x, "gray") for x in sentiment_counts.index], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    ax.set_xticklabels(sentiment_order, rotation=45)
    st.pyplot(fig2)



    # Risk Level Distribution
    st.subheader("Risk Level Distribution")
    risk_order = ["Low", "Medium", "High"]
    risk_colors = {
        "Low": "yellow",
        "Medium": "orange",
        "High": "red"
    }
    risk_counts = df["risk_level"].value_counts().reindex(risk_order).fillna(0)
    fig3, ax = plt.subplots()
    risk_counts.plot(kind="bar", color=[risk_colors.get(x, "gray") for x in risk_counts.index], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Risk Level Distribution")
    ax.set_xticklabels(risk_order, rotation=0)
    st.pyplot(fig3)



    # Distribution of tweets per topic (pie chart)
    st.subheader("Tweet Distribution per Topic")
    topic_counts = df["assigned_topic"].value_counts()
    fig4, ax = plt.subplots()
    ax.pie(topic_counts, labels=topic_counts.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig4)


    # Proportion of sentiment per topic (stacked bar plot)
    st.subheader("Proportion of Sentiment per Topic")
    desired_order = ['very positive', 'positive', 'negative', 'very negative']
    sentiment_topic = pd.crosstab(
        df["assigned_topic"],
        df["sentiment_label"],
        normalize="index"
        )
    sentiment_topic = sentiment_topic.reindex(columns=desired_order).fillna(0)
    fig5, ax = plt.subplots(figsize=(10, 5))
    sentiment_topic.plot(
        kind="bar",
        stacked=True,
        color=[sentiment_colors.get(x, "gray") for x in sentiment_topic.columns],
        ax=ax
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Sentiment Proportion by Topic")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig5)



    # Risk level distribution by location (top 5 locations)
    st.subheader("Risk Level Distribution by Location (Top 5 Locations)")
    df['Location'] = df['Location'].fillna('Unknown')
    # Count number of tweets per location
    location_counts = df['Location'].value_counts().reset_index(name='count')
    location_counts.columns = ['Location', 'count']
    # Get top 5 locations
    top_5_locations = location_counts.head(5)['Location']
    # Filter data to include only top 5 locations
    top_5_data = df[df['Location'].isin(top_5_locations)]
    # Count of risk levels per location (excluding Unknown)
    risk_counts = top_5_data[top_5_data['Location'] != 'Unknown'] \
        .groupby(['Location', 'risk_level']) \
        .size().reset_index(name='count')
    # Sort locations by total tweet count
    location_order = risk_counts.groupby('Location')['count'] \
        .sum().sort_values(ascending=False).index
    risk_palette = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    fig6, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=risk_counts,
        y='Location',
        x='count',
        hue='risk_level',
        palette=risk_palette,
        order=location_order,
        ax=ax
    )
    ax.set_title("Risk Level Distribution by Location (Top 5 Locations)")
    ax.set_xlabel("Number of Tweets")
    ax.set_ylabel("Location")
    ax.legend(title="Risk Level", loc="upper right")
    plt.tight_layout()

    st.pyplot(fig6)



    # Monthly Average Risk Score Over Time
    st.subheader("📈 Monthly Average Risk Score Over Time")
    df['YearMonth'] = df['Datetime'].dt.to_period('M')
    monthly_avg_risk = df.groupby('YearMonth')['risk_score'].mean().reset_index()
    monthly_avg_risk['YearMonth'] = monthly_avg_risk['YearMonth'].dt.to_timestamp()
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=monthly_avg_risk,
        x='YearMonth',
        y='risk_score',
        marker='o',
        linewidth=2,
        color='purple',
        ax=ax7
    )
    ax7.set_xlabel("Month")
    ax7.set_ylabel("Average Risk Score")
    ax7.set_title("Monthly Average Risk Score Over Time")
    ax7.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig7)


# --- Sentiment Analysis Page ---
def sentiment_page():
    st.markdown("<h2 style='text-align: center; color: green;'>🌾 🇲🇾 Food Security Sentiment Analysis 🌾</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='
            background-color: #f0f8ff; 
            padding: 10px; 
            border-radius: 10px; 
            border-left: 5px solid #1f77b4;
            margin-bottom: 10px;
        '>
            <p style='font-size: 14px; color: #333;'>
                Enter your text on the left or load a sample tweet.<br>
                Our AI model will predict if the sentiment is 
                <b style='color:green;'>Positive</b> or 
                <b style='color:red;'>Negative</b> with a confidence score.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # Load model and pipeline
    text_pipeline = load_text_pipeline()
    model, tokenizer = load_model_tokenizer()

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📝 Input Tweet")

        if st.button("Load Sample Tweet"):
            st.session_state.input_text = random.choice(sample_tweets)

        user_input = st.text_area("Enter tweet text here:", height=200, key="input_text")

        predict_clicked = st.button("🎯 Predict Sentiment")

    with col2:
        st.markdown("### 🔍 Prediction Results")  # Always show this title

        if predict_clicked:
            if not user_input.strip():
                st.warning("Please enter some text to predict!")
                return

            input_texts = user_input.strip().split("\n")
            cleaned_texts = text_pipeline.transform(input_texts)

            encoded = tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=94,
                return_tensors="pt"
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            label_map = {0: "Negative", 1: "Positive"}

            for i, text in enumerate(input_texts):
                st.markdown(f"**Original:** {text}")
                st.markdown(f"**Cleaned:** {cleaned_texts[i]}")

                pred_label = label_map.get(preds[i], "Unknown")
                conf_score = probs[i][preds[i]]
                color = "green" if pred_label == "Positive" else "red"

                st.markdown(f"**Prediction:** <span style='color:{color}; font-weight:bold;'>{pred_label}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {conf_score:.2%}")

                # Prepare data for seaborn
                df_conf = pd.DataFrame({
                    'Sentiment': ["Negative", "Positive"],
                    'Confidence': probs[i],
                    'Color': ["red", "green"]
                })

                fig, ax = plt.subplots(figsize=(6, 0.6))
                sns.barplot(
                    x="Confidence",
                    y="Sentiment",
                    data=df_conf,
                    palette=df_conf["Color"].tolist(),
                    ax=ax,
                    orient="h"
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence Score")
                ax.set_title("Confidence Scores")
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No prediction yet. Enter a tweet and click '🎯 Predict Sentiment' to see results.")



    


# --- Main app ---
def main():
    st.set_page_config(page_title="Malaysia Food Security Sentiment Analysis", page_icon="🌾", layout="centered")

    # --- Custom CSS Styling ---
    st.markdown("""
        <style>
       
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        .footer {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-top: 20px;
        }
        .footer:hover {
            color: #2ca02c;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    # --- Sidebar Navigation ---
    with st.sidebar:
        # Custom header with icon and gradient text
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 0px;">
                <img src="https://cdn.iconscout.com/icon/premium/png-256-thumb/seasonal-food-insecurity-5373665-4485818.png" width="80" />
                <h3 style="margin-bottom:0; background: -webkit-linear-gradient(#228B22, #32CD32); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Food Security App
                </h3>
                 <p style="font-size:13px; color:#666;">AI-powered insight for social good</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.title("🌐  Navigation")
        page = st.radio("Choose a Page", ["📊 EDA Dataset", "🤖 Sentiment Analysis"], index=1)

        st.markdown("---")
        st.markdown("""<small style="color:gray;">Powered by AI & Data</small>""", unsafe_allow_html=True)
        

    # --- Page Routing ---
    if "EDA" in page:
        eda_page()
    elif "Sentiment" in page:
        sentiment_page()

    # --- Footer ---
    st.markdown("""
        <div class='footer'>
            Made with ❤️ by <b>Ibtihel</b> | Powered by <i>DistilBERT</i>, <i>Streamlit</i> & <i>Seaborn</i> 🍲
        </div>
    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
