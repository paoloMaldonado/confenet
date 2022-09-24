# Bar plots
# Qualitative validation of the model

import matplotlib.pyplot as plt
import math

def plot_keyword(model, df, keywords):
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
    plot_values = []
    for keyword in keywords:
        df["Indexes"] = df.token_sentence.str.find(keyword)
        total = df[df.Indexes > 0].shape[0]
        x = ["cluster\n "+str(i) for i in range(model.topics)] 
        y = []
        for i in range(model.topics):
            cluster_shape = df.Indexes[(model.cluster_method.labels_ == i) & (df.Indexes > 0)].shape[0]
            y.append(cluster_shape)

        plot_values.append([x, y])

    if len(keywords) == 1:
        fig, ax = plt.subplots(figsize =(8, 5))
        x, y = plot_values[0]
        ax.bar(x, y)
        # TODO: quitar (%) y reemplazar en latex
        ax.set_ylabel('cantidad (%)')
        ax.set_title('Keyword: '+ keyword)
    else:
        # fig, axs = plt.subplots(2, 2, figsize =(8, 3))
        # for i, (ax, keyword) in enumerate(zip(axs, keywords)):
        #     x, y = plot_values[i]
        #     ax.bar(x, y)
        #     if i == 0:
        #         ax.set_ylabel('cantidad (%)')
        #     ax.set_title('Keyword: '+ keyword)
        # keyword = "multi"

        grid_x = math.ceil(len(keywords)/2)
        fig, axs = plt.subplots(grid_x, 2, figsize =(8, 8))
        axs = axs.ravel()
        for i, (ax, keyword) in enumerate(zip(axs, keywords)):
            x, y = plot_values[i]
            ax.bar(x, y)
            if i % 2 == 0:
                ax.set_ylabel('cantidad (%)')
            ax.set_title('Keyword: '+ keyword)
        keyword = "multi"
    
    plt.savefig(model.method + '_' + str(model.topics) + '_topics_' + keyword +'.png')
    plt.close()