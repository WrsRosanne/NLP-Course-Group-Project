import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
nltk.download('stopwords')

def stem_lyrics(df, lyrics_column='Lyrics', new_column='lyrics_stemmed'):
    """
    Stems the lyrics and adds as a new column.
    
    INPUT: 
        df: pandas DataFrame
        lyrics_column: name of column containing lyrics
        new_column: name for new stemmed column
    RETURNS: DataFrame with new stemmed lyrics column
    """
    stemmer = PorterStemmer()
    
    def stem_text(text):
        words = word_tokenize(text.lower())
        stemmed = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed)
    
    df[new_column] = df[lyrics_column].apply(stem_text)
    return df

