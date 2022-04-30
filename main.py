import pandas as pd
import numpy as np
from model.model import *
from model.utils import *
from model.ctf_idf import *
import pprint
from sklearn.feature_extraction.text import CountVectorizer

#### main ####
if __name__ == "__main__":
    df = pd.read_pickle('data/confesiones_allclean_new.df')
    data = list(df.no_stopwords)

    model = Model(topics=5, method='word2vec', epochs=30)
    model.fit(posts=df.post_clean, token_list=data, dimension_output=4000)

    docs = pd.DataFrame({'Document': df.token_sentence, 'Class': model.cluster_method.labels_})
    docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})
    
    # Create bag of words
    count_vectorizer = CountVectorizer().fit(docs_per_class.Document)
    count = count_vectorizer.transform(docs_per_class.Document)
    words = count_vectorizer.get_feature_names()

    # Extract top 10 words per class
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()
    topics = np.unique(model.cluster_method.labels_)
    words_per_class = {'topic' + str(topics[label]): [words[index] for index in ctfidf[label].argsort()[-20:]] 
                    for label in docs_per_class.Class}
    
    print(words_per_class, end=" ")