import pandas as pd
import numpy as np
from model.model import *
from model.utils import *
from model.validate import *
from model.boew import BOEW
import pprint
from tqdm import tqdm
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#### main ####
if __name__ == "__main__":
    plt.style.use("seaborn")
    df = pd.read_pickle('data/espol_clean.df')
    data = list(df.no_stopwords)

    pretrained_path = "pretrained/all_countries/word2vec_confesionesInternacional.wordvectors"

    model = Model(topics=4, method='word2vec', epochs=30)
    model.load(pretrained_path, 'word2vec')
    model.fit(posts=df.post_clean, token_list=data, dimension_output=300, dimension_reduction_method="pca", num_components=18)

    plot_keyword(model, df, "politica")

    # 4 topics - pca 2 dimensions
    # dim = "2D"
    # model = Model(topics=4, method='fasttext', epochs=30)
    # model.load(pretrained_path, "fasttext")
    # model.fit(posts=df.post_clean, token_list=data, dimension_output=200,
    #           dimension_reduction_method="boew", reduction_vectorizer="fasttext", num_components=2, pretrained_path=pretrained_path)

    # 4 topics - pca 18 dimensions
    # dim = "18D"
    # model = Model(topics=4, method='word2vec', epochs=30)
    # model.load(pretrained_path, "word2vec")
    # model.fit(posts=df.post_clean, token_list=data, dimension_output=200,
    #           dimension_reduction_method="pca", num_components=18, pretrained_path=pretrained_path)

    # 6 topics - no pca
    # dim = "200D"
    # model = Model(topics=6, method='word2vec', epochs=30)
    # model.load(pretrained_path, "word2vec")
    # model.fit(posts=df.post_clean, token_list=data, dimension_output=200)


    #plt.style.use("seaborn")
    # print(model.x_features.shape)

    # keywords_academico = ["cursos", "ciclo", "examenes", "profesor", "virtuales", "clases"]
    # keywords_politica = ["castillo", "elecciones", "keiko", "politica", "votar", "voto"]
    # keywords_social = ["amigos", "amigas", "casa", "fiesta", "salir", "buscar", "interesado"]
    # keywords_amor = ["relacion", "flaca", "novio", "novia", "ex", "extraño"]
    # keywords = [keywords_academico, keywords_politica, keywords_social, keywords_amor]
    # names = ["academico", "politica", "vida social", "amor"]

    # import os

    # os.chdir("validation_plots")
    # for k_list, cat in zip(keywords, names):
    #     os.mkdir(cat)
    #     os.chdir(cat)
    #     os.mkdir(dim)
    #     os.chdir(dim)
    #     for keyword in k_list:
    #         plot_keyword(model, df, keyword)
    #     os.chdir("..")
    #     os.chdir("..")

    # # sillouethe
    #print(get_silhouette(model))

    # # # # # plots
    #get_inertia_plot(model)

    #visualize_clusters(model, labels=model.cluster_method.labels_)
    #visualize_clusters_3D(model, labels=model.cluster_method.labels_)

    # # # word clouds
    # # print("regular")
    # # print(get_topic_words(token_list=data, token_sentence=df.token_sentence, labels=model.cluster_method.labels_))
    # print("topics")
    #print(get_topic_words(token_list=data, token_sentence=df.token_sentence, labels=model.cluster_method.labels_, c_tfidf=True, num_topics=10))

    # # # # get_word_clouds(model, labels=model.cluster_method.labels_, token_sentence=df.token_sentence, token_list=data)
    #get_word_clouds(model, labels=model.cluster_method.labels_, token_sentence=df.token_sentence, token_list=data, c_tfidf=True)

    # metrics = ["c_v", "u_mass"]

    # for m in metrics:
    #     avg_coherence = 0
    #     coherence = get_coherence(model, token_list=data, token_sentence=df.token_sentence, metric=m, c_tfidf=True, num_topics=100)
    #     print(m+": "+str(coherence))

    # scores = []
    # for t in range(4, 14+1):
    #     model = Model(topics=t, method='mpnet', epochs=30)
    #     model.load(pretrained_path, "mpnet")
    #     model.fit(posts=df.post_clean, token_list=data, dimension_output=300)
    #     scores.append(get_coherence(model, token_list=data, token_sentence=df.token_sentence, metric='c_v', c_tfidf=True, num_topics=100))

    # scores_even = scores[::2]

    # get_inertia_plot(model)

    # plt.figure(figsize=(10,10))
    # plt.plot(range(4, 14+1), scores, 'bx-')
    # plt.plot(range(4, 14+1, 2), scores_even, 'rx-')
    # plt.xlabel('Values of K')
    # plt.ylabel('CV Coherence score')
    # plt.title('Coherence versus number of topics')
    # plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + 'cv_coherence.png')


    ## LDA Experiment ###
    # scores = []
    # for t in range(2, 20+1, 2):
    #     model = Model(topics=t, method='LDA')
    #     model.fit(posts=df.post_clean, token_list=data)
    #     scores.append(get_coherence(model, token_list=data, token_sentence=df.token_sentence))


    # plt.figure(figsize=(10,10))
    # plt.plot(range(2, 20+1, 2), scores, 'bx-')
    # plt.xticks(np.arange(2, 20+1, step=2))
    # plt.xlabel('Values of K')
    # plt.ylabel('CV Coherence score')
    # plt.title('Coherence versus number of topics')
    # plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + 'cv_coherence.png')