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
    return df

def punctuation(df):
    df['question_count'] = df['Lyrics'].apply(lambda x: x.count('?'))
    df['exclamation_count'] = df['Lyrics'].apply(lambda x: x.count('!'))
    return df

def inertia_plot(X_scaled):
    inertia_values = []
    k_values = [] # storage for k values so we can easily plot and visualize elbow
    for k in range(2, 40):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=15) # 15 runs for sparse, high dimensional data
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_)
        k_values.append(k)
    plt.figure(figsize=(5, 5))
    plt.plot(k_values, inertia_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.ylim(bottom=0)
    plt.show()

def cluster_model(cluster_df, X_scaled, n):
    k_model = KMeans(n_clusters=n, random_state=42, n_init=15) # 15 runs for sparse, high dimensional data
    k_model.fit(X_scaled)
    results = k_model.predict(X_scaled)
    cluster_df['k_cluster'] = results
    return cluster_df

def rank_years(fit_df):
    max_year = fit_df['Year'].max()
    min_year = fit_df['Year'].min()
    print(f"Max year: {max_year}")
    print(f"Min year: {min_year}")
    
    # print our results grouped by year for each cluster
    print(fit_df.groupby('k_cluster')['Year'].mean().sort_values())

def visualize_variables(df1, str1, str2):
    data_subset = df1[df1['k_cluster'].isin([0, 7])]
    plt.figure(figsize=(5, 5))
    for cluster in [0, 7]:
        cluster_data = data_subset[data_subset['k_cluster'] == cluster]
        plt.scatter(cluster_data[str1],
                    cluster_data[str2],
                    label=f'Cluster {cluster}',alpha=0.5)
    
    plt.xlabel(str1)
    plt.ylabel(str2)
    plt.title(str1 + ' vs ' + str2 + '(Clusters 0 & 7)')
    plt.show()