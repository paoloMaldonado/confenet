# Class Model
# Main class

import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from model.boew import BOEW
from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
    """
    Instanciate a Model object

    parameters:
        topics: int. default 5. number of k topics the model will find
        method: str. {'word2vec', 'doc2vec', 'fasttext', 'mpnet', ,'tfidf', 'LDA'}. default 'word2vec'. 
        epochs. int. default 30
    """
    def __init__(self, topics=5, method='word2vec', epochs=30):
        """
        Constructor function

        parameters description:
            topics: int. number of k topics the model will find
            method: str. vectorizer method.
            cluster_method: KMeans object instance. clustering method 
            LDA_model: LDA object instance. LDA method
            epochs: int. number of epochs that the vectorizers will take for training
            word_vectors: KeyedVectors object. A dictionary that contains the words with their respective vector representations
            doc_vectors: KeyedVectors object.  A dictionary that contains the documents (sentences) with their respective vector representations
            x_features: ndarray. A numpy array of sentence vectors
            dictionary. Dictionary object. A dictionary between words with their respective ids
            corpus. list. A set of non-repeated words that appears in the whole dataset
            is_loaded. bool. An indicator used for checking if a pretrained vectorizer model is already loaded
            project_features. bool. If True, then a dimensional reduction technique (PCA, BOEW) will be perfomed by the Model instance
        """
        self.topics = topics
        
        self.method = method
        self.cluster_method = None
        self.LDA_model = None
        
        self.epochs = epochs
        self.word_vectors = None
        self.doc_vectors = None
        self.x_features = None

        self.dictionary = None
        self.corpus = None

        self.is_loaded = False
        self.project_features = False
    
    def vectorize(self, posts, token_list, dimension_output, method=None, min_count=2, window=5):
        """
        vectorize the data using a word/sentence embedding model
        returns: ndarray
            a numpy array of sentence vectors 

        parameters:
            posts: pandas, list. posts already preprocessed at a sentence level 
            token_list: list. tokenized clean documents
            dimension_ouput: int. desired dimension of the ouput feature vector (not for LDA and others 
                                  pretrained models)   
            method: str. {'word2vec', 'doc2vec', 'fasttext', 'mpnet', 'tfidf', 'LDA'}. embedding/bow vectorizer 
            min_count: int. default 2. take the words whose ocurrences across all documents is greater than min_count   
            window: int. default 5. for text embeddings
        """
        if method == None:
            method = self.method
        
        if method == 'word2vec':
            if self.is_loaded:

                print("Creating sentence vectors")
                X = self.get_matrix_features(token_list, self.word_vectors)
                #print(X.shape)
                print("word2vec has finished")

            else:
                print("Starting word2vec vectorizer")
                model_cbow = Word2Vec(min_count = min_count, vector_size= dimension_output, window = window, sg=0, epochs=self.epochs, workers=6) 
                model_cbow.build_vocab(token_list)
                print("word2vec vectorizer is already running")
                model_cbow.train(token_list, total_examples=model_cbow.corpus_count, epochs=model_cbow.epochs)
                
                self.word_vectors = model_cbow.wv
                X = self.get_matrix_features(token_list, self.word_vectors)
                    
                print("word2vec has finished")
        
        elif method == 'doc2vec':

            if self.is_loaded:
                sentence_vectors = self.doc_vectors.dv
                doc2vec_array = []
                for i in range(len(sentence_vectors)):
                    doc2vec_array.append(sentence_vectors[i])
                X = np.asarray(doc2vec_array)
                
                print("doc2vec has finished")

            else:
                print("Starting doc2vec vectorizer")
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(token_list)]
                model_dbow = Doc2Vec(vector_size = dimension_output, min_count = min_count, window = 5, epochs=self.epochs)
                model_dbow.build_vocab(documents)
                print("doc2vec vectorizer is already running")
                model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
                
                self.doc_vectors = model_dbow

                doc2vec_array = []
                for i in range(len(model_dbow.dv)):
                    doc2vec_array.append(model_dbow.dv[i])
                X = np.asarray(doc2vec_array)
                
                print("doc2vec has finished")
        
        elif method == 'fasttext':
            if self.is_loaded:
                print("Creating sentence vectors")
                X = self.get_matrix_features(token_list, self.word_vectors)
                print("fasttext has finished")
            else:
                print("Starting fastText vectorizer")
                model = FastText(vector_size=dimension_output, window=5, min_count=min_count, epochs=self.epochs)
                model.build_vocab(token_list)
                print("fastText vectorizer is already running")
                model.train(token_list, total_examples=len(token_list), epochs=model.epochs)
                
                self.word_vectors = model.wv
                X = self.get_matrix_features(token_list, self.word_vectors)
                
                print("fastText has finished")
        
        elif method == 'mpnet':
            if self.is_loaded:
                print("Creating sentence vectors")
                X = self.x_features
                print("mpnet has finished loading the matrix of features")
            else:
                print("Starting mpnet vectorizer")
                model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                print("mpnet vectorizer is already running")
                sentence_embeddings = model.encode(posts, show_progress_bar=True)
                
                X = sentence_embeddings.copy()
                
                print("mpnet has finished")
        
        elif method == "tfidf":
            if self.is_loaded:
                print("Creating sentence vectors")
                X = self.x_features
                print("TF-IDF has finished loading the matrix of features")
            else:
                vectorizer = TfidfVectorizer(min_df=10, tokenizer = lambda x:x, preprocessor= lambda x:x, token_pattern=None)
                X = vectorizer.fit_transform(token_list)
                X = X.toarray()

                print("TF-IDF has finished")
            
        return X
        
    def get_sentence_vector(self, word_vectors, tokenized_sentence):
        """
        get the sentence vector out of a list of word vectors 
        returns: ndarray. a numpy array of a single sentence vector
                 None. if there is no word vectors for a given sentence

        parameters:
            word_vectors. KeyedVectors. a dictionary that contains the words with their respective vector representations  
            tokenized_sentence. list. clean document tokenized 
        """
        doc = [word for word in tokenized_sentence if word in word_vectors]
        if len(doc) > 0:
            return np.mean(normalize(word_vectors[doc]), axis=0)  # normalize vector before averaging
        else:
            return None

    def get_matrix_features(self, token_list, word_vectors):
        """
        get the matrix of sentences vectors
        returns ndarray.
            A multidimensional numpy array (matrix) that contains all the documents in vector form   
        
        parameters:
            token_list. list. tokenized clean documents
            word_vectors. KeyedVectors. a dictionary that contains the words with their respective vector representations
        """

        X = [dv for doc in token_list if (dv := self.get_sentence_vector(word_vectors, doc)) is not None]
        X = np.asarray(X)
        return X
    
    def fit(self, posts, token_list, dimension_output=4000, 
            cluster_method=None, dimension_reduction_method=None, reduction_vectorizer="word2vec", 
            num_components=2, pretrained_path=None, **kwargs):
        """
        fit the data into a clustering method or LDA
        returns:
            An already trained clustering model object or LDA object

        parameters:
            posts: pandas, list. posts already preprocessed at a sentence level 
            token_list: list. tokenized and clean posts
            dimension_ouput: int. desired dimension of the ouput feature vector (not for LDA and others 
                                  pretrained models)   
            cluster_method: KMeans object. default KMeans instance. clustering model
            dimension_reduction_method. str. {'pca', 'boew'}. dimension reduction model 
            reduction_vectorizer. str. default 'word2vec'. {'word2vec', 'fasttext'}. vectorizer used by the boew model
            num_components. int. default 2. number of desired dimensions (only if a reduction method is defined)
            pretrained_path. str. file path of the pretrained vectorizer model for boew
            kwargs: gensim models args
                min_count: str. 
                window: str.    
        """    

        if cluster_method is None:
            cluster_method = KMeans
        
        if not self.dictionary:
           self.dictionary = Dictionary(token_list)
           self.corpus = [self.dictionary.doc2bow(text) for text in token_list]

        if self.method == 'LDA':            
            print('Starting and running LDA model')
            self.LDA_model = LdaModel(corpus=self.corpus,
                                      id2word=self.dictionary,
                                      num_topics=self.topics, 
                                      random_state=100,
                                      chunksize=1000,
                                      passes=10,
                                      alpha='auto')
            
            print('LDA model has finished')
        else:
            self.cluster_method = cluster_method(n_clusters=self.topics ,random_state=10)
            
            if 'min_count' in kwargs:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method, min_count=kwargs['min_count'])
            elif 'window' in kwargs:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method, window=kwargs['window'])
            else:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method)

            print('Starting Clustering')

            if dimension_reduction_method == "pca":
                print("projecting data using PCA...")
                self.project_features = True
                pca = PCA(n_components=num_components, random_state=10)
                self.x_features = pca.fit_transform(self.x_features)

            elif dimension_reduction_method == "boew":
                print("projecting data using BOEW...")
                self.project_features = True
                boew = BOEW(n=num_components, embedding_method_name=reduction_vectorizer, random_state=10)
                boew.load_vectorizer(pretrained_path, reduction_vectorizer)
                self.x_features = boew.fit_transform(token_list)
                

            self.cluster_method.fit(self.x_features)
            print('Clustering Done!')

    def save(self):
        """
        Save the vectorizer model
        returns:
            a file with the pretrained vectorizer
        """
        if self.method == 'doc2vec':
            print("Saving doc2vec model...")
            self.doc_vectors.save("doc2vec_confesionesPeru.docvectors")
            del self.doc_vectors
            print("Model saved...")

        elif self.method == 'LDA':
            print("LDA")

        elif self.method == 'mpnet':
            print("Saving mpnet model features...")
            np.save("mpnet_features.npy", self.x_features)
            print("Matrix saved...")

        elif self.method == 'word2vec':
            print("Saving word2vec model...")
            self.word_vectors.save("word2vec_confesionesPeru.wordvectors")
            print("Model saved...")

        elif self.method == 'fasttext':
            print("Saving fasttext model...")
            self.word_vectors.save("fasttext_confesionesPeru.wordvectors")
            print("Model saved...")

        else:
            print("Unknown model")

    
    def load(self, model_path, method):
        """
        Load a pretrained vectorizer model
        parameters:
            model_path: str. Path to pretrained vectorizer model file
            method: str. {'word2vec', 'doc2vec', 'fasttext', 'mpnet', 'tfidf', 'LDA'}. Vectorizer method name 
        """
        if method == 'doc2vec':
            model_d2v = Doc2Vec.load(model_path)
            self.doc_vectors = model_d2v
            self.method = method
            self.is_loaded = True
            print("Loaded doc2vec model...")

        elif method == 'LDA':
            print("LDA")

        elif method == 'mpnet':
            self.x_features = np.load(model_path)
            self.method = method
            self.is_loaded = True
            print("Loaded mpnet features...")

        elif method == 'word2vec':
            wv = KeyedVectors.load(model_path, mmap='r')
            self.word_vectors = wv
            self.method = method
            self.is_loaded = True
            print("Loaded word2vec model...")
        
        elif method == 'fasttext':
            wv = KeyedVectors.load(model_path, mmap='r')
            self.word_vectors = wv
            self.method = method
            self.is_loaded = True
            print("Loaded fasttext model...")

