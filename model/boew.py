# BOEW Class
# Implementation of the Bag Of Embedding Words technique for dimension reduction

import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

class BOEW:
    """
    Instanciate a BOEW object

    parameters:
        n: int. Number of dimensions of the ouput frequency vector
        embeding_method_name: str. {'word2vec', 'fasttext'}. default 'word2vec'. embedding method 
        word_vectors: KeyedVectors object. A dictionary that contains the words with their respective vector representations
        word_to_cluster: dict. A python dictionary that contains a word and the number of cluster it belongs to (word:cluster_label)
        is_loaded: bool. If true, then a pretrained model was loaded
        random_state: int. default 10. The seed used for the KMeans algorithm
    """
    def __init__(self, n, embedding_method_name='word2vec', word_vectors=None, word_to_cluster=None, is_loaded=False, random_state=10):
        self.n = n
        self.embedding_method_name = embedding_method_name
        self.word_vectors = word_vectors
        self.is_loaded = is_loaded

        if word_to_cluster == None:
            self.word_to_cluster = {}
        else:
            self.word_to_cluster = word_to_cluster
        
        self.random_state = random_state
    
    def fit(self, X):
        """
        applies vectorization and kmeans clustering 
        returns: BOEW instance
            An instance of BOEW class with updated member attributes

        parameters:
            X: list. input tokenized data
        """
        vectorizer_model = None
        if self.embedding_method_name == None:
            print("No vectorizer model has been initialized")
            return

        # if no pretained model was loaded
        if self.is_loaded == False:
            if self.embedding_method_name == "word2vec":
                vectorizer_model = Word2Vec
            elif self.embedding_method_name == "fasttext":
                vectorizer_model = FastText
            else:
                print("Error, document-vectorizers are not compatible")
                return

            vectorizer = vectorizer_model(min_count = 2, vector_size = 200, window = 5, epochs=30, workers=6) 
            vectorizer.build_vocab(X)
            vectorizer.train(X, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
        
            self.word_vectors = vectorizer.wv
        
        # create matrix of word vectors
        bag_of_embeddings = []
        for word in self.word_vectors.index_to_key:
            bag_of_embeddings.append(self.word_vectors.get_vector(word))
        
        bag_of_embeddings = np.asarray(bag_of_embeddings)

        # cluster the word corpus
        kmeans = KMeans(n_clusters=self.n, random_state=self.random_state)
        cluster_method = kmeans.fit(bag_of_embeddings)

        # create a dictionary of word:cluster#
        for word, cluster in zip(self.word_vectors.index_to_key, cluster_method.labels_):
            self.word_to_cluster[word] = cluster

        # returns a new updated instance
        return BOEW(n=self.n, embedding_method_name=self.embedding_method_name, 
                    word_vectors=self.word_vectors, word_to_cluster=self.word_to_cluster, is_loaded=self.is_loaded)

    def transform(self, X):
        """
        creates the matrix of frequencies of k columns
        return: ndarray
            matrix of frequencies 

        parameters:
            X: list. input tokenized data
        """
        frequency_matrix = []
        for document in X:
            # creates a n-dimensional vector of zeros 
            frequency_vector = np.zeros(shape=self.n)
            for word in document:
                # if word is present in the dictionary, then
                # add 1 to the frequency vector
                if word in self.word_to_cluster.keys():
                    frequency_vector[self.word_to_cluster[word]] += 1
            # append the frequency vector to the frequency matrix
            frequency_matrix.append(frequency_vector)

        # casto to numpy array
        frequency_matrix = np.asarray(frequency_matrix)
        # Standarize the values
        x_transform = StandardScaler().fit_transform(frequency_matrix)
        return x_transform

    def fit_transform(self, X):
        """
        perform kmeans on a vectorized corpus and then create the matrix of frequencies of k columns
        returns: ndarray
            matrix of frequencies
        
        parameters:
            X: list. input tokenized data
        """
        return self.fit(X).transform(X)
        
    
    def load_vectorizer(self, model_path, method):
        """
        Load a pretrained vectorizer model
        parameters:
            model_path: str. Path to pretrained vectorizer model file
            method: str. {'word2vec', 'fasttext'}. Vectorizer method name 
        """
        if method == 'word2vec':
            wv = KeyedVectors.load(model_path, mmap='r')
            self.word_vectors = wv
            self.embedding_method_name = method
            self.is_loaded = True
            print("Loaded word2vec model...")
        
        elif method == 'fasttext':
            wv = KeyedVectors.load(model_path, mmap='r')
            self.word_vectors = wv
            self.embedding_method_name = method
            self.is_loaded = True
            print("Loaded fasttext model...")

