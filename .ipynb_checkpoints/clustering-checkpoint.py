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

def tf_idf(df):
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english',token_pattern=r'\b[a-zA-Z]{3,}\b')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df