import pandas as pd
import numpy as np
from gensim import corpora, similarities
from gensim import models as gensim_models
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import os.path

class LDATransformer(TransformerMixin):

    def __init__(self, num_topics=20, tfidf=False, 
                 passes=3, iterations=700, min_prob=0, min_freq=0, save_dict=False, dict_file=None, random_seed=None):
        
        # should be tuned
        self.num_topics = num_topics
        self.tfidf = tfidf
        
        # may be left as default
        self.passes = passes
        self.iterations = iterations
        self.min_prob = min_prob
        self.min_freq = min_freq
        
        # for reproducability
        self.random_seed = random_seed
        self.save_dict = save_dict
        self.dict_file = dict_file
        
        self.dictionary = None
        self.lda_model = None
        self.tfidf_model = None
        

        
    def fit(self, X):
        if self.dict_file is not None and os.path.isfile(self.dict_file) :
            self.dictionary = corpora.Dictionary.load(self.dict_file)
        else:
            self.dictionary = self._generate_dictionary(X)
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
    
    
    def _generate_dictionary(self, X):
        data = X.values.flatten('F')
        texts = [[word for word in str(document).lower().split()] for document in data]
        dictionary = corpora.Dictionary(texts)
        if self.save_dict:
            dictionary.save(self.dict_file)
        return dictionary
    
    
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
    


class PVTransformer(TransformerMixin):

    def __init__(self, size=16, window=8, min_count=1, workers=1, alpha=0.025, dm=1, epochs=10, random_seed=None):
        
        self.random_seed = random_seed
        self.pv_model = None
        
        # should be tuned
        self.size = size
        self.window = window
        self.alpha = alpha
        self.dm = dm
        
        # may be left as default
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        
        
    def fit(self, X):
        train_comments = X.values.flatten('F')
        train_documents = self._generate_labeled_sentences(train_comments)
        
        self.pv_model = Doc2Vec(size=self.size, window=self.window, alpha=self.alpha, min_count=self.min_count, workers=self.workers, seed=self.random_seed, dm=self.dm) 
        self.pv_model.build_vocab(train_documents)
        np.random.seed(self.random_seed)
        for epoch in range(self.epochs):
            np.random.shuffle(train_documents)
            self.pv_model.train(train_documents)
            
        return self

    
    def fit_transform(self, X):
        self.fit(X)
        
        nrow = X.shape[0]
        ncol = X.shape[1]
        
        train_X = self.pv_model.docvecs[range(nrow*ncol)]
        train_X = np.hstack(np.vsplit(train_X, ncol))
        colnames = ["pv%s_event%s"%(vec+1, event+1) for event in range(ncol) for vec in range(self.size)]

        train_X = pd.DataFrame(train_X, columns=colnames, index=X.index)
        return train_X
    
    
    def transform(self, X):
        ncol = X.shape[1]
            
        test_comments = X.values.flatten('F')
        vecs = [self.pv_model.infer_vector(comment.split()) for comment in test_comments]
        test_X = np.hstack(np.vsplit(np.array(vecs), ncol))
        colnames = ["pv%s_event%s"%(vec+1, event+1) for event in range(ncol) for vec in range(self.size)]
        
        test_X = pd.DataFrame(test_X, columns=colnames, index=X.index)
        test_X.to_csv("test_X_pv2.csv", sep=";")
        return test_X
    
    
    def _generate_labeled_sentences(self, comments):
        documents = [LabeledSentence(words=comment.split(), tags=[i]) for i, comment in enumerate(comments)]
        return(documents)
    
    

class BoNGTransformer(TransformerMixin):

    def __init__(self, ngram_min=1, ngram_max=1, tfidf=False, nr_selected=100):
        
        # should be tuned
        self.ngram_max = ngram_max
        self.tfidf = tfidf
        self.nr_selected = nr_selected
        
        # may be left as default
        self.ngram_min = ngram_min
        
        self.vectorizer = None
        self.feature_selector = SelectKBest(chi2, k=self.nr_selected)
        self.selected_cols = None
        
        
    def fit(self, X, y):
        data = X.values.flatten('F')
        if self.tfidf:
            self.vectorizer = TfidfVectorizer(ngram_range=(self.ngram_min,self.ngram_max))
        else:
            self.vectorizer = CountVectorizer(ngram_range=(self.ngram_min,self.ngram_max))
        bong = self.vectorizer.fit_transform(data)

        # select features
        if self.nr_selected=="all" or len(self.vectorizer.get_feature_names()) <= self.nr_selected:
            self.feature_selector = SelectKBest(chi2, k="all")
        self.feature_selector.fit(bong, y)
        
        # remember selected column names
        if self.nr_selected=="all":
            self.selected_cols = np.array(self.vectorizer.get_feature_names())
        else:
            self.selected_cols = np.array(self.vectorizer.get_feature_names())[self.feature_selector.scores_.argsort()[-self.nr_selected:][::-1]]
        
        return self
    
    
    def transform(self, X):
        bong = self.vectorizer.transform(X)
        bong = self.feature_selector.transform(bong)
        bong = bong.toarray()
        
        return pd.DataFrame(bong, columns=self.selected_cols, index=X.index)
    
    
    
class NBLogCountRatioTransformer(TransformerMixin):

    def __init__(self, ngram_min=1, ngram_max=1, alpha=1.0, nr_selected=100, pos_label="positive"):
        
        # should be tuned
        self.ngram_max = ngram_max
        self.alpha = alpha
        self.nr_selected = nr_selected
        
        # may be left as default
        self.ngram_min = ngram_min
        
        self.pos_label = pos_label
        self.vectorizer = CountVectorizer(ngram_range=(ngram_min,ngram_max))
        
        
    def fit(self, X, y):
        data = X.values.flatten('F')
        bong = self.vectorizer.fit_transform(X)
        bong = bong.toarray()
        
        # calculate nb ratios
        pos_bong = bong[y == self.pos_label]
        neg_bong = bong[y != self.pos_label]
        pos_sum = pos_bong.sum()
        neg_sum = neg_bong.sum()
        p = 1.0 * pos_bong.sum(axis=0) + self.alpha
        q = 1.0 * neg_bong.sum(axis=0) + self.alpha
        r = np.log((p / pos_sum) / (q / neg_sum))
        self.nb_r = r
        r = np.squeeze(np.asarray(r))
        
        # feature selection
        if (self.nr_selected >= len(r)): 
            r_selected = range(len(r))
        else:
            r_sorted = np.argsort(r)
            r_selected = np.concatenate([r_sorted[:self.nr_selected/2], r_sorted[-self.nr_selected/2:]])
        self.r_selected = r_selected
        
        if self.nr_selected=="all":
            self.selected_cols = np.array(self.vectorizer.get_feature_names())
        else:
            self.selected_cols = np.array(self.vectorizer.get_feature_names())[self.r_selected]
            
        return self
    
    
    def transform(self, X):
        bong = self.vectorizer.transform(X)
        bong = bong.toarray()
            
        # generate transformed selected data
        bong = bong * self.nb_r
        bong = bong[:,self.r_selected]
        
        return pd.DataFrame(bong, columns=self.selected_cols, index=X.index)
    
    