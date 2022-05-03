import numpy as np
from collections import Counter
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pprint
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

def get_topic_words(token_list, labels, k=None):
    """
    get top words within each topic from clustering results
    returns:
        a list of words for each cluster
    parameters:
        token_list: list. tokenized and clean posts
        labels: ndarray. labels of each point
    """
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
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics

def get_coherence(model, token_list, metric='c_v'):
    """
    get the coherence value of the model
    returns:
        A float representing the coherence value
    parameters:
        model: Model object. An instance of Model
        metric. str. default 'c_v'. Specify the metric of coherence
    """
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.LDA_model, texts=token_list, corpus=model.corpus, dictionary=model.dictionary, coherence=metric)
    else:
        topics = get_topic_words(token_list, model.cluster_method.labels_)
        cm = CoherenceModel(topics=topics, texts=token_list, corpus=model.corpus, dictionary=model.dictionary, coherence=metric)
    
    return cm.get_coherence()

def get_silhouette(model):
    if model.method != 'LDA': # if method is not LDA
        print("Getting silhouette score...")
        sc = silhouette_score(model.x_features, model.cluster_method.labels_)
        return sc
    else:
        print("not supported by the current model")
        return None

def get_inertia_plot(model):
    if model.method == 'LDA':
        print("LDA model does not support this function")
        return None
    else:
        print("Getting plot of inertias...")
        scores = []
        n_clusters = range(2, 15, 2)
        for k in n_clusters:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(model.x_features)
            scores.append(kmeans.inertia_)

        plt.figure(figsize=(10,10))
        plt.plot(n_clusters, scores, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + 'inertias.png')


def visualize_data(model):
    if model.method == 'LDA':
        print("LDA model does not support this function")
        return None
    else:   
        clusterable_embedding = umap.UMAP(
            # n_neighbors=30,
            # min_dist=0.0,
            # n_components=2,
            random_state=42,
            transform_seed=40
        ).fit_transform(model.x_features)

        plt.figure(figsize=(10,10))
        plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], cmap='Spectral')
        plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + '_umap_2d_visualization.png')

def visualize_clusters(model, labels=None):
    if model.method == 'LDA':
        print("LDA model does not support this function")
        return None
    else:
        clusterable_embedding = umap.UMAP(
            # n_neighbors=30,
            # min_dist=0.0,
            # n_components=2,
            random_state=42,
            transform_seed=40
        ).fit_transform(model.x_features)

        u_labels = np.unique(labels)
        plt.figure(figsize=(10,10))
        for i in u_labels:
            plt.scatter(clusterable_embedding[labels == i, 0] , clusterable_embedding[labels == i, 1] , label = i, cmap='Spectral')
        plt.legend()
        plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + '_cluster_visualization.png')

def get_word_clouds(model, labels=None, token_sentence=None):
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
            wordcloud = WordCloud(background_color="white", max_words=500).generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig('topics/' + model.method + '_cluster_' +  str(label_num) + '.png')

        print("The wordclouds has been saved in the topics folder")