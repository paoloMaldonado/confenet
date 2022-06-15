# Bar plots
# Qualitative validation of the model

import matplotlib.pyplot as plt

def plot_keyword(model, df, keyword):
    """
    get a bar plot of the frequency of a specific keyword across each topic,
    this function is used for the qualititive analysis of the results
    returns:
        A png file of the bar plot (the file will be exported to the topics folder)
    
    parameters:
        model: Model object. an instance of Model class
        df: pandas dataframe object. the whole dataset in pandas format
        keyword: str. The specific keyword
    """
    df["Indexes"] = df.token_sentence.str.find(keyword)
    total = df[df.Indexes > 0].shape[0]
    x = ["cluster\n "+str(i) for i in range(model.topics)] 
    y = []
    for i in range(model.topics):
        cluster_shape = df.Indexes[(model.cluster_method.labels_ == i) & (df.Indexes > 0)].shape[0]
        y.append(cluster_shape)
    plt.bar(x, y)
    plt.xlabel('clusters')
    plt.ylabel('cantidad (%)')
    plt.title('Keyword: '+ keyword)
    plt.savefig(model.method + '_' + str(model.topics) + '_topics_' + keyword +'.png')
    plt.close()