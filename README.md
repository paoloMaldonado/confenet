# Natural Language Processing Techniques for Behavior Analysis in Social Networks of Hispanic American University Communities
Repository of our research project that uses text embedding algorithms and clustering techniques to find relevant topics among student's confessions communities on Facebook and Instagram. Data are available for any research or academic purposes in the *complete_dataset* folder. 

### Example usage
Run the model and print 10 words per topic: 
```sh
python main.py --get-topics 10 
```
### Required libraries
In order to run the code, the following libraries are required:
- numpy >= 1.21.5
- gensim >= 4.1.2
- scikit-learn >= 1.0.2
- sentence_transformers >= 2.2.0
- matplotlib >= 3.5.1
- pandas >= 1.3.4
- scipy >= 1.7.2

### Optional arguments
Additionally, the code provides optional console parameters for getting specific plots such as the clusters plot, wordclouds or the frequency bar plot of a certain keyword. The full list of optional arguments is listed down below:

| Argument | Description |
| ----------- | ----------- |
| ```-d``` | path/to/dataframe |
| ```-m``` | specify the vectorizer method to be performed by the model|
| ```-t``` | number of topics to find by the model |
| ```--dimension-output``` | vector dimension of the vectorizer ouput|
| ```--vectorizer-path``` | path/to/pretrained/vectorizer |
| ```--reduction-method``` | specify the reduction method to be performed by the model |
| ```--boew-vectorizer``` | specify the vectorizer method for boew |
| ```--num-components``` | specify the number of components after reduction process|
| ```--boew-pretrained``` | path/to/pretained/boew-vectorizer/ |
| ```--get-topics``` | print the specified number of words per topic |
| ```--disable-c-tfidf``` | disable c-tf*idf |
| ```--silhouette``` | print the silhouette score of the model |
| ```--plot-keyword``` | get a bar plot of the frequency of a keywords (separated by comma) across each topic |
| ```--plot-inertia``` | get a inertia plot of the model |
| ```--plot-clusters``` | get a 2D plot of the clusters  |
| ```--get-word-clouds``` | get a plot of the wordclouds for each cluster |
| ```--get-coherence-metrics``` | print the coherence metrics of the model |


