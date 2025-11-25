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
nltk.download('stopwords')


def load_in():
    """
    Loads in raw dataset for song lyrics as provided from the online source.

    INPUT(S): None

    RETURNS: pandas DF from CSV file in local folder
    """
    df = pd.read_csv("dataset/all_songs_data_processed.csv")
    return df

def remove_links(df_input):
    """
    Removes all columns that contain links / API output / redundant values 
    to simplify the analysis.

    INPUT(S): pandas dataframe

    RETURNS: pandas DF with link columns removed
    """
    return df_input[['Song Title', 'Artist', 'Lyrics', 'Year']]

def remove_nan(df_input):
    """
    Removes NaN values of release Date, Lyrics, and Year from dataframe.

    INPUT(S): pandas dataframe

    RETURNS: pandas DF with nan values removed
    """
    return df_input.dropna(subset=['Lyrics', 'Year'])

def clean_data(df_input):
    """

    """
    df_1 = remove_links(df_input).copy()
    df_2 = remove_nan(df_1).copy()
    return df_2



