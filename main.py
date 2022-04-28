import pandas as pd
import numpy as np
from model.model import *

#### main ####
if __name__ == "__main__":
    df = pd.read_pickle('data/confesiones_allclean_new.df')
    data = list(df.no_stopwords)

    model = Model(topics=5, method='word2vec', epochs=30)
    model.fit(posts=df.post_clean, token_list=data, dimension_output=100)

