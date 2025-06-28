import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SpatialDropout1D
from wordcloud import WordCloud
import seaborn as sns
import nltk
import string
from textblob import TextBlob
import pickle

# Load the trained model and tokenizer


@st.cache_resource
def load_sentiment_model():
    model = load_model('models/model.h5')
    return model


@st.cache_resource
def load_tokenizer():
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('models/sentiment_label.pkl', 'rb') as f:
        sentiment_label = pickle.load(f)
    return tokenizer, sentiment_label


model = load_sentiment_model()
tokenizer, sentiment_label = load_tokenizer()

st.title('US Airline Twitter Sentiment Analysis')
st.write('Predict the sentiment (positive/negative) of US airline tweets using a deep learning model and TextBlob.')

# Sidebar for navigation
option = st.sidebar.selectbox(
    'Choose an option:', ['Predict Sentiment', 'Visualizations'])

if option == 'Predict Sentiment':
    st.header('Predict Sentiment of a Tweet')
    user_input = st.text_area('Enter a tweet:')
    if st.button('Predict'):
        if user_input:
            # LSTM Model Prediction
            tw = tokenizer.texts_to_sequences([user_input])
            tw = pad_sequences(tw, maxlen=200)
            pred = model.predict(tw)
            prediction = int(np.round(pred[0][0]))
            lstm_label = sentiment_label[1][prediction]

            # TextBlob Prediction
            def clean_tweet(tweet):
                tweet = ''.join(
                    ch for ch in tweet if ch not in string.punctuation)
                stopwords = nltk.corpus.stopwords.words('english')
                tweet = ' '.join(
                    [word for word in tweet.split() if word not in stopwords])
                return tweet

            def get_sentiment(tweet):
                analysis = TextBlob(tweet)
                polarity = analysis.sentiment.polarity
                return 'positive' if polarity >= 0 else 'negative'
            nltk.download('stopwords', quiet=True)
            cleaned = clean_tweet(user_input)
            textblob_label = get_sentiment(cleaned)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('LSTM Model')
                st.success(f'Sentiment: {lstm_label}')
            with col2:
                st.subheader('TextBlob')
                st.success(f'Sentiment: {textblob_label}')
            st.toast("Prediction complete!", icon="🎉")
            st.balloons()
        else:
            st.warning('Please enter a tweet.')
    st.write('**Accuracy of the model is 96.5%**')

elif option == 'Visualizations':
    st.header('Data Visualizations')
    st.subheader('Sentiment Ratio Pie Chart')
    st.image('images/pie chart.png', caption='Sentiment Ratio Pie Chart', use_column_width=True)

    st.subheader('Negative Sentiment WordCloud')
    st.image('images/negative words wordcloud.png', caption='Negative Sentiment WordCloud', use_column_width=True)

    st.subheader('Positive Sentiment WordCloud')
    st.image('images/positive words wordcloud.png', caption='Positive Sentiment WordCloud', use_column_width=True)

    st.subheader('Confusion Matrix')
    st.image('images/confusion matrix.png', caption='Confusion Matrix', use_column_width=True)

    st.subheader('Accuracy vs Validation Accuracy')
    st.image('images/accuracy.png', caption='Accuracy vs Validation Accuracy', use_column_width=True)

    st.subheader('Loss vs Validation Loss')
    st.image('images/loss.png', caption='Loss vs Validation Loss', use_column_width=True)

