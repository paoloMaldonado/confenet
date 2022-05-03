import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class Model:
    """
        Instanciate a Model object

        parameters:
            topics: int. default 5. number of k topics the model will find
            method: str. {'word2vec', 'doc2vec', 'fasttext', 'mpnet', 'LDA'}. default 'word2vec'. 
            epochs. int. default 30
    """
    def __init__(self, topics=5, method='word2vec', epochs=30):
        self.topics = topics
        
        self.method = method
        self.cluster_method = None
        self.LDA_model = None
        
        self.epochs = epochs
        self.word_vectors = None
        self.x_features = None

        self.dictionary = None
        self.corpus = None
    
    def vectorize(self, posts, token_list, dimension_output, method=None, min_count=2, window=5):
        """
        vectorize the data using a word/sentence embedding model
        returns: ndarray
            a numpy array of sentence vectors 

        parameters:
            posts: pandas, list. posts already preprocessed at a sentence level 
            token_list: list. tokenized and clean posts
            dimension_ouput: int. desired dimension of the ouput feature vector (not for LDA and others 
                                  pretrained models)   
            method: str. {'word2vec', 'doc2vec', 'fasttext', 'mpnet', 'LDA'}. embedding model 
            min_count: int. default 2
            window: int. default 5
        """
        if method == None:
            method = self.method
        
        if method == 'word2vec':
            print("Starting word2vec vectorizer")
            model_cbow = Word2Vec(min_count = min_count, vector_size= dimension_output, window = window, sg=0, epochs=self.epochs) 
            model_cbow.build_vocab(token_list)
            print("word2vec vectorizer is already running")
            model_cbow.train(token_list, total_examples=model_cbow.corpus_count, epochs=model_cbow.epochs)
            
            self.word_vectors = model_cbow.wv
            X = [dv for doc in token_list if (dv := self.get_sentence_vector(self.word_vectors, doc)) is not None]
            X = np.asarray(X)
                
            print("word2vec has finished")
        
        elif method == 'doc2vec':
            print("Starting doc2vec vectorizer")
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(token_list)]
            model_dbow = Doc2Vec(vector_size = dimension_output, min_count = min_count, window = 5, epochs=self.epochs)
            model_dbow.build_vocab(documents)
            print("doc2vec vectorizer is already running")
            model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
            
            doc2vec_array = []
            for i in range(len(model_dbow.dv)):
                doc2vec_array.append(model_dbow.dv[i])
            X = np.asarray(doc2vec_array)
            
            print("doc2vec has finished")
        
        elif method == 'fasttext':
            print("Starting fastText vectorizer")
            model = FastText(vector_size=dimension_output, window=5, min_count=min_count, epochs=self.epochs)
            model.build_vocab(token_list)
            print("fastText vectorizer is already running")
            model.train(token_list, total_examples=len(token_list), epochs=model.epochs)
            
            self.word_vectors = model.wv
            X = [dv for doc in token_list if (dv := self.get_sentence_vector(self.word_vectors, doc)) is not None]
            X = np.asarray(X)
            
            print("fastText has finished")
        
        elif method == 'mpnet':
            print("Starting mpnet vectorizer")
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            print("mpnet vectorizer is already running")
            sentence_embeddings = model.encode(posts, show_progress_bar=True)
            
            X = sentence_embeddings.copy()
            
            print("mpnet has finished")
            
        return X
        
    def get_sentence_vector(self, word_vectors, tokenized_sentence):
        doc = [word for word in tokenized_sentence if word in word_vectors]
        if len(doc) > 0:
            return np.mean(normalize(word_vectors[doc]), axis=0)  # normalize vector before averaging
        else:
            return None
    
    def fit(self, posts, token_list, dimension_output, cluster_method=None, **kwargs):
        """
        fit the data into a clustering method or LDA
        returns: 
            An already trained clustering model object or LDA object

        parameters:
            posts: pandas, list. posts already preprocessed at a sentence level 
            token_list: list. tokenized and clean posts
            dimension_ouput: int. desired dimension of the ouput feature vector (not for LDA and others 
                                  pretrained models)   
            cluster_method: str. default 'kmeans'. clustering model
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
                                      passes=10)
            
            print('LDA model has finished')
        else:
            self.cluster_method = cluster_method(n_clusters=self.topics)
            
            if 'min_count' in kwargs:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method, min_count=kwargs['min_count'])
            elif 'window' in kwargs:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method, window=kwargs['window'])
            else:
                self.x_features = self.vectorize(posts, token_list, dimension_output, self.method)

            print('Starting Clustering')
            self.cluster_method.fit(self.x_features)
            print('Clustering Done!')