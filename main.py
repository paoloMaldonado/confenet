import pandas as pd
from model.model import *
from model.utils import *
from model.validate import *
from model.boew import BOEW

import argparse
 
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt

#### main ####
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Adding optional argument for setting up the model
    parser.add_argument("-d", "--data-path", default = "data/confesiones_peru_clean.df", help= "path/to/dataframe")
    parser.add_argument("-m", "--vectorizer", default="word2vec")
    parser.add_argument("-t", "--topics", default=4, type=int)
    parser.add_argument("--dimension-output", default=300, type=int)
    parser.add_argument("--vectorizer-path", default = "pretrained/word2vec/word2vec_confesionesPeru.wordvectors", help= "path/to/vectorizer")
    parser.add_argument("--reduction-method", choices=["pca", "boew", "None"], default="pca")
    parser.add_argument("--boew-vectorizer", choices=["word2vec", "fasttext"], default="word2vec")
    parser.add_argument("--num-components", default=18, type=int)
    parser.add_argument("--boew-pretrained", default = "pretrained/word2vec/word2vec_confesionesPeru.wordvectors")

    # Adding optional arguments for additional functionality
    parser.add_argument("--get-topics", default=10, type=int)
    parser.add_argument("--disable-c-tfidf", action="store_true")
    parser.add_argument("--silhouette", action="store_true")
    parser.add_argument("--plot-keyword", action="store_true")
    parser.add_argument("--plot-inertia", action="store_true")
    parser.add_argument("--plot-clusters", action="store_true")
    parser.add_argument("--plot-clusters-3d", action="store_true")
    parser.add_argument("--get-word-clouds", action="store_true")
    parser.add_argument("--get-coherence-metrics", action="store_true")

    args = vars(parser.parse_args())

    # Parameters for model
    path_to_data = args["data_path"]
    topics_number = args["topics"]
    vectorizer_method = args["vectorizer"]
    pretrained_path = args["vectorizer_path"]
    dimension_ouput = args["dimension_output"]
    reduction_method = args["reduction_method"]
    boew_vectorizer = args["boew_vectorizer"]
    n_components = args["num_components"]
    boew_pretrained = args["boew_pretrained"]

    # Parameters for functionality
    words_per_topic = args["get_topics"] 

    # matplotlib style
    plt.style.use("seaborn")

    # load the data
    df = pd.read_pickle(path_to_data)
    data = list(df.no_stopwords)

    if reduction_method == "None":
        reduction_method = None

    #print(args)

    # training the model
    model = Model(topics=topics_number, method=vectorizer_method, epochs=30)
    model.load(pretrained_path, method=vectorizer_method)
    model.fit(posts=df.post_clean, 
              token_list=data, 
              dimension_output=dimension_ouput, 
              dimension_reduction_method=reduction_method,
              reduction_vectorizer=boew_vectorizer,
              num_components=n_components,
              pretrained_path=boew_pretrained)

    ### plot keywords
    if args["plot_keyword"]:
        plot_keyword(model, df, "politica")

    ### silhouette
    if args["silhouette"]:
        print(get_silhouette(model))

    ### plot inretias
    if args["plot_inertia"]:
        get_inertia_plot(model)
    
    ### visualize clusters
    if args["plot_clusters"]:
        visualize_clusters(model, labels=model.cluster_method.labels_)

    ### visualize 3D clusters
    if args["plot_clusters_3d"]:
        visualize_clusters_3D(model, labels=model.cluster_method.labels_)

    ### disable c-tfidf
    use_c_tfidf = True
    if args["disable_c_tfidf"]:
        use_c_tfidf = False

    ### print topics
    if args["get_topics"]:
        print(get_topic_words(token_list=data, token_sentence=df.token_sentence, 
                              labels=model.cluster_method.labels_, c_tfidf=use_c_tfidf, num_topics=words_per_topic))

    ### get wordclouds
    if args["get_word_clouds"]:
        get_word_clouds(model, labels=model.cluster_method.labels_, token_sentence=df.token_sentence, 
                        token_list=data, c_tfidf=use_c_tfidf)

    ### get coherence metrics
    if args["get_coherence_metrics"]:
        metrics = ["c_v", "u_mass"]
        for m in metrics:
            avg_coherence = 0
            coherence = get_coherence(model, token_list=data, token_sentence=df.token_sentence, metric=m, 
                                      c_tfidf=use_c_tfidf, num_topics=100)
            print(m+": "+str(coherence))


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


    #visualize_clusters_3D(model, labels=model.cluster_method.labels_)

    # scores = []
    # for t in range(4, 14+1):
    #     model = Model(topics=t, method='mpnet', epochs=30)
    #     model.load(pretrained_path, "mpnet")
    #     model.fit(posts=df.post_clean, token_list=data, dimension_output=300)
    #     scores.append(get_coherence(model, token_list=data, token_sentence=df.token_sentence, metric='c_v', c_tfidf=True, num_topics=100))

    # scores_even = scores[::2]

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