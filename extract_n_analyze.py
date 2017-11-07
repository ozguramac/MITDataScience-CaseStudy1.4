path_to_mitie_lib = 'libmitie.so'
path_to_ner_model = 'MITIE-models/english/ner_model.dat'

import sys
sys.path.append(path_to_mitie_lib)

import numpy as np

from mitie import named_entity_extractor
from mitie import tokenize
from goose import Goose
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer

#prepare data refs
urls = []
urls.append('https://www.nytimes.com/2017/11/06/world/asia/trump-xi-jinping-visit-china.html')
urls.append('https://www.politico.com/story/2017/11/07/jeff-sessions-george-papadopoulos-russia-questions-244639')
urls.append('http://www.wbur.org/onpoint/2017/11/07/dnc-chair-tom-perez-rigged-is-a-loaded-dangerous-term')
urls.append('http://www.businessinsider.com/snapchat-reports-q3-earnings-2017-11')
urls.append('http://www.cnn.com/2017/11/06/politics/carter-page-testimony-released/index.html')
urls.append('https://www.bloomberg.com/news/articles/2017-11-06/house-tax-chief-targets-carried-interest-in-bill-state-of-play')
urls.append('https://www.thenation.com/article/turkey-is-sliding-deeper-into-dictatorship/')
urls.append('https://www.bloomberg.com/news/articles/2017-11-07/turkey-u-s-visa-progress-set-back-by-new-sparring-over-trials')
urls.append('https://www.nytimes.com/2017/11/07/technology/waymo-autonomous-cars.html')
urls.append('https://www.cnbc.com/2017/11/06/feds-williams-lays-out-a-case-for-switch-in-how-fed-sets-rates.html')

#init extraction tool
g = Goose()

# total number of articles to process
N = len(urls)

# in memory stores for the topics, titles and contents of the news stories
topics_array = []
titles_array = []
corpus = []
for i in range(0, N):
    # extract article
    article = g.extract(url=urls[i])
    # get the contents of the article
    corpus.append(article.cleaned_text.encode('utf-8').replace('\n','').lower())
    #get the original topic of the article
    topics_array.append(article.meta_description.encode('utf-8').replace('\n','').lower())
    #get the title of the article
    titles_array.append(article.title.encode('utf-8').replace('\n','').lower())

# load ner model
ner = named_entity_extractor(path_to_ner_model)

# entity subset array
entity_text_array = []
for i in range(0, N):
    # Load the article corpus and convert it into a list of words.
    tokens = tokenize(corpus[i])
    # extract all entities known to the ner model mentioned in this article
    entities = ner.extract_entities(tokens)

    # extract the actual entity words and append to the array
    for e in entities:
        range_array = e[0]
        tag = e[1]
        if len(e) > 2:
            score = e[2]
            score_text = "{:0.3f}".format(score)
        entity_text = " ".join(tokens[j] for j in range_array)
        entity_text_array.append(entity_text.lower())

# remove duplicate entities detected
entity_text_array = np.unique(entity_text_array)

# The following lines of code can help represent each article in the dataset as a vector of TF-IDF values
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=entity_text_array)
corpus_tf_idf = vect.fit_transform(corpus)

# change n_clusters to equal the number of clusters desired
n_clusters = 7
n_components = n_clusters

#spectral clustering
spectral = cluster.SpectralClustering(n_clusters= n_clusters, eigen_solver='arpack', affinity="nearest_neighbors", n_neighbors = 10)
spectral.fit(corpus_tf_idf)

# output in the following format (one line per article)

print 'article_number, topic, spectral_clustering_cluster_number, article_title'
print '------------------------------------------------------------------------'

if hasattr(spectral, 'labels_'):
    cluster_assignments = spectral.labels_.astype(np.int)
for i in range(0, len(cluster_assignments)):
    print (i, topics_array[i], cluster_assignments [i], titles_array[i])