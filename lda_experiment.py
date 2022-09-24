import pandas as pd
from model.model import *
from model.utils import *
from model.validate import *

if __name__ == "__main__":
    df = pd.read_pickle("data/confesiones_peru_clean.df")
    data = list(df.no_stopwords)

    ## LDA Experiment ###
    scores = []
    for t in range(2, 20+1, 2):
        model = Model(topics=t, method='LDA')
        model.fit(posts=df.post_clean, token_list=data)
        scores.append(get_coherence(model, token_list=data, token_sentence=df.token_sentence))

    plt.figure(figsize=(8,6))
    plt.plot(range(2, 20+1, 2), scores, 'bx-')
    plt.xticks(np.arange(2, 20+1, step=2))
    plt.xlabel('Valores de K')
    plt.ylabel('CV Coherence score')
    plt.title('Coherence vs numero de topicos')
    plt.savefig('plots/' + model.method + '_' + str(model.topics) + '_topics_' + 'cv_coherence.eps')