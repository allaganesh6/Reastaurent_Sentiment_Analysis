
import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Load the trained model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_artifacts()

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Identify the grammatical category of a word
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)

# Perform lemmatization
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = []

    for word in words:
        pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos)
        lemmatized_words.append(lemma)

    return ' '.join(lemmatized_words)

# Preprocess the user's review
def preprocess_review(text):
    text = text.lower()

    punctuation_table = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_table)

    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    text = lemmatize_text(text)

    return text

# Streamlit application interface
st.set_page_config(
    page_title="Restaurant Sentiment Analysis",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write(
    "Enter a restaurant review and discover "
    "whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: The food was fantastic!"
)

if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        processed_review = preprocess_review(review)

        if not processed_review:
            st.warning("No usable text remains after preprocessing.")

        else:
            review_vectorized = vectorizer.transform(
                [processed_review]
            )

            prediction = model.predict(review_vectorized)[0]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            with st.expander("View processed review"):
                st.write(processed_review)
                
