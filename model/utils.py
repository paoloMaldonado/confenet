# Utilities class
# Get plots and scores related to Model

import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pprint
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from model.ctf_idf import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_topic_words(token_list, token_sentence, labels, num_topics=10, k=None, c_tfidf=False):
    """
    based on Stveshawn' Contextual topic identification project
    https://github.com/Stveshawn/contextual_topic_identification

    get top words within each topic from clustering results
    returns: list
        a list of words for each cluster
    parameters:
        token_list: list. tokenized and clean posts
        token_sentence: pandas column, list. clean sentences  
        labels: ndarray. labels of each point
        num_topics: int. default 10. top-n words in each topic given either by c-tfidf or frequencies
        k: int. number of unique labels
        c_tfidf: bool. True if c-tfidf will be used 
    """
    if c_tfidf == False:
        if k is None:
            k = len(np.unique(labels))
        topics = ['' for _ in range(k)]
        for i, c in enumerate(token_list):
            # bag of words for every topic
            topics[labels[i]] += (' ' + ' '.join(c))
        # count the number of words in every topic bag, returns a list of word - number of ocurrence for every topic
        word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
        # get sorted word counts
        word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
        # get topics
        topics = list(map(lambda x: list(map(lambda x: x[0], x[:num_topics])), word_counts))
    else:
        docs = pd.DataFrame({'Document': token_sentence, 'Class': labels})
        docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})
        
        # Create bag of words
        count_vectorizer = CountVectorizer().fit(docs_per_class.Document)
        count = count_vectorizer.transform(docs_per_class.Document)
        words = count_vectorizer.get_feature_names_out()

        # Extract top 10 words per class
        ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()
        topics = [[words[index] for index in ctfidf[label].argsort()[-num_topics:]] for label in docs_per_class.Class]

    return topics

def get_coherence(model, token_list, token_sentence, metric='c_v', c_tfidf=False, num_topics=10):
    """
    get the coherence value of the model
    returns: float
        A float representing the coherence value
    parameters:
        model: Model object. An instance of Model class
        token_list: list. tokenized clean documents
        token_sentence: pandas column, list. clean sentences
        metric: str. default 'c_v'. {'c_v', 'u_mass'} . Specify the metric of coherence  
        c_tfidf: bool. True if c-tfidf will be used
        num_topics: int. default 10. top-n words in each topic given either by c-tfidf or frequencies 

        token. str. default 'c_v'. Specify the metric of coherence
    """
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.LDA_model, texts=token_list, corpus=model.corpus, dictionary=model.dictionary, coherence=metric, topn=100)
    else:
        topics = get_topic_words(token_list, token_sentence, model.cluster_method.labels_, c_tfidf=c_tfidf, num_topics=num_topics)
        cm = CoherenceModel(topics=topics, texts=token_list, corpus=model.corpus, dictionary=model.dictionary, coherence=metric)
    
    return cm.get_coherence()

def get_silhouette(model):
    """
    get the clustering silhouette score
    return: float 
        A float representing the silhouette value
    
    parameters:
        model: Model object. An instance of Model class
    """
    if model.method != 'LDA': # if method is not LDA
        print("Getting silhouette score...")
        sc = silhouette_score(model.x_features, model.cluster_method.labels_)
        return sc
    else:
        print("not supported by the current model")
        return None

def get_inertia_plot(model):
    """
    get a plot of inertia vs k clusters
    returns: 
        A png file of the plot (the file will be exported to the plots folder)
    
    parameters:
        model: Model object. An instance of Model class
    """
    if model.method == 'LDA':
        print("LDA model does not support this function")
        return None
    else:
        print("Getting plot of inertias...")
        scores = []
        n_clusters = range(2, 15, 2)
        for k in n_clusters:
            kmeans = KMeans(n_clusters=k, random_state=10)
            kmeans.fit(model.x_features)
            scores.append(kmeans.inertia_)

        plt.figure(figsize=(10,10))
        plt.plot(n_clusters, scores, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + 'inertias.png')


# def visualize_data(model):
#     if model.method == 'LDA':
#         print("LDA model does not support this function")
#         return None
#     else:   
#         clusterable_embedding = umap.UMAP(
#             # n_neighbors=30,
#             # min_dist=0.0,
#             # n_components=2,
#             random_state=42,
#             transform_seed=40
#         ).fit_transform(model.x_features)

#         plt.figure(figsize=(10,10))
#         plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], cmap='Spectral')
#         plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + '_umap_2d_visualization.png')

def visualize_clusters(model, labels=None):
    """
    get a plot of the clusters
    returns: 
        A png file of the plot (the file will be exported to the plots folder)
    
    parameters:
        model: Model object. an instance of Model class
        labels: list. label of each sentence/post after the clustering process
    """
    if model.method == 'LDA':
        print("LDA model does not support this function")
        return None
    else:
        clusterable_embedding = model.x_features
        centroids = np.array(model.cluster_method.cluster_centers_)

        if clusterable_embedding.shape[1] > 2:
            print("plotting...")
            # project the features into a 2-dimensional space
            pca = PCA(2, random_state=10)
            clusterable_embedding = pca.fit_transform(clusterable_embedding)
            centroids = pca.fit_transform(centroids)

        u_labels = np.unique(labels)
        plt.figure(figsize=(10,10))
        for i in u_labels:
            plt.scatter(clusterable_embedding[labels == i, 0] , clusterable_embedding[labels == i, 1] , label = i, cmap='Spectral')
            
        plt.scatter(centroids[:,0], centroids[:,1], s = 80, marker='+', color = 'k')
    
        plt.legend()
        plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + '_cluster_visualization.png')


def visualize_clusters_3D(model, labels=None):    
    clusterable_embedding = model.x_features
    centroids = np.array(model.cluster_method.cluster_centers_)

    # project the features into a 3-dimensional space
    pca = PCA(3, random_state=10)
    clusterable_embedding = pca.fit_transform(clusterable_embedding)
    centroids = pca.fit_transform(centroids)

    u_labels = np.unique(labels)
    plt.figure(figsize=(10,10))


    plt.rcParams["figure.figsize"] = (25,15) 
    fig = plt.figure(1) 
    plt.clf() 
    ax = Axes3D(fig, 
                rect = [0, 0, .95, 1], 
                elev = 20, 
                azim = 134) 
    plt.cla() 

    for i in u_labels:
        ax.scatter(clusterable_embedding[labels == i, 0] , clusterable_embedding[labels == i, 1], clusterable_embedding[labels == i, 2],
                    s = 200,
                    cmap = 'spring',
                    alpha = 0.5,
                    edgecolor = 'darkgrey')
    
    ax.set_xlabel('x', fontsize = 16)
    ax.set_ylabel('y', fontsize = 16) 
    ax.set_zlabel('z', fontsize = 16) 
    plt.show() 

    # plt.scatter(clusterable_embedding[labels == i, 0] , clusterable_embedding[labels == i, 1] , label = i, cmap='Spectral')
    # plt.scatter(centroids[:,0], centroids[:,1], s = 80, marker='+', color = 'k')

    #plt.legend()
    #plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + '_cluster_visualization.png')

def get_word_clouds(model, labels=None, token_sentence=None, token_list=None ,c_tfidf=False):
    """
    get a plot of the wordclouds for each cluster
    return:
        Several png files of the wordclouds (the file will be exported to the topics folder)
    
    parameters:
        model: Model object. an instance of Model class
        labels: list. label of each sentence/post after the clustering process
        token_sentence: pandas column, list. clean sentences
        token_list: list. tokenized clean documents
        c_tfidf: bool. True if c-tfidf will be used for getting the words of each topic
    """
    if c_tfidf == False:
        if model.method == 'LDA':
            print("Getting wordclouds of clusters...")
            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

            cloud = WordCloud(background_color='white',
                            max_words=100,
                            colormap='tab10',
                            prefer_horizontal=1.0)

            topics = model.LDA_model.show_topics(num_topics=model.topics ,num_words=100, formatted=False)
            
            print("Generating wordclouds...")
            for label, _ in enumerate(topics):
                plt.figure()
                topic_words = dict(topics[label][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.imshow(cloud)
                plt.axis('off')
                plt.savefig('topics/' + model.method + '_cluster_' +  str(label) + '.png')

            print("The wordclouds has been saved in the topics folder")
        else:
            print("Getting wordclouds of clusters...")
            u_labels = np.unique(labels)

            for label_num in u_labels:
                cluster = list(token_sentence[labels == label_num])
                text = ','.join(cluster)
                wordcloud = WordCloud(background_color="white", max_words=1000).generate(text)
                plt.figure()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.savefig('topics/' + model.method + '_cluster_' +  str(label_num) + '.png')

            print("The wordclouds has been saved in the topics folder")
    else:
        print("Getting c-tfidf wordclouds of clusters...")
        words_per_class = get_topic_words(token_list, token_sentence, labels, num_topics=30, c_tfidf=c_tfidf)
        words_per_class = words_per_class[::-1]
        u_labels = np.unique(labels)

        for label in u_labels:
            text = " ".join(words_per_class[label])
            wordcloud = WordCloud(max_font_size=40, background_color="white").generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig('topics/' + model.method + '_ctfidf+cluster_' +  str(label) + '.png')
        print("The wordclouds has been saved in the topics folder")