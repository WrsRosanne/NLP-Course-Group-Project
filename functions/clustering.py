import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import re
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def tf_idf(df, n):
    vectorizer = TfidfVectorizer(max_features=n, stop_words='english',token_pattern=r'\b[a-zA-Z]{3,}\b')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

def get_sentiment(lyrics):
    blob = TextBlob(lyrics)
    return blob.sentiment.polarity  # Returns -1 (negative) to 1 (positive)

def append_sent_and_subj(df):
    df['sentiment'] = df['Lyrics'].apply(get_sentiment)
    df['subjectivity'] = df['Lyrics'].apply(lambda x: TextBlob(x).sentiment.subjectivity)