import pandas as pd
import numpy as np
from gensim import corpora, similarities
from gensim import models as gensim_models
from sklearn.base import TransformerMixin

class LDATransformer(TransformerMixin):

    def __init__(self, dictionary, num_topics=20, tfidf=False, 
                 passes=3, iterations=700, min_prob=0, min_freq=0, random_seed=None):
        
        self.random_seed = random_seed
        
        # should be pre-built
        self.dictionary = dictionary
        
        # should be tuned
        self.num_topics = num_topics
        self.tfidf = tfidf
        
        # may be left as default
        self.passes = passes
        self.iterations = iterations
        self.min_prob = min_prob
        self.min_freq = min_freq
        
        self.lda_model = None
        self.tfidf_model = None
        

        
    def fit(self, X):
        corpus = self._generate_corpus_data(X)
        np.random.seed(self.random_seed)
        self.lda_model = gensim_models.LdaModel(corpus, id2word=self.dictionary, num_topics=self.num_topics, 
                                                passes=self.passes, iterations=self.iterations, minimum_probability=self.min_prob)
        return self

    
    def transform(self, X):
        ncol = X.shape[1]
        corpus = self._generate_corpus_data(X)
        topics = self.lda_model[corpus]
        topic_data = np.zeros((len(topics), self.num_topics))
        for i in range(len(topics)):
            for (idx, prob) in topics[i]:
                topic_data[i,idx] = prob
        topic_data = np.hstack(np.vsplit(topic_data, ncol))
        topic_colnames = ["topic%s_event%s"%(topic+1, event+1) for event in range(ncol) for topic in range(self.num_topics)]

        return pd.DataFrame(topic_data, columns=topic_colnames, index=X.index)
    
    
    def _generate_corpus_data(self, X):
        data = X.values.flatten('F')
        texts = [[word for word in str(document).lower().split()] for document in data]
        
        # if frequency threshold set, filter
        if self.min_freq > 0:
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > self.min_freq] for text in texts]
        
        # construct corpus
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # if requested, do tfidf transformation
        if self.tfidf:
            if self.tfidf_model == None:
                self.tfidf_model = gensim_models.TfidfModel(corpus)
            corpus_tfidf = self.tfidf_model[corpus]
            return(corpus_tfidf)
        return corpus