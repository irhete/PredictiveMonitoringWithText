# PredictiveMonitoringWithText
Scripts related to predictive business process monitoring framework with structured and unstructured (textual) data.


## Dependencies

* python 3.5
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [gensim](https://radimrehurek.com/gensim/) (for LDA and doc2vec models)
* [estnltk](https://github.com/estnltk/estnltk) (for lemmatization in Estonian language)



## Getting started


## Preprocessing

Before using the text models, textual data should be lemmatized. The example below constructs a list of lemmatized documents `docs_as_lemmas`, given a list of raw documents (`corpus`), using a lemmatizer for Estonian language.

```python
from estnltk import Text

docs_as_lemmas = []
for document in corpus:
    text = Text(document.lower())
    docs_as_lemmas.append(" ".join(text.lemmas))

```
    

## Text models

The text models are implemented as custom transformers, which include `fit`, `transform`, and `fit_transform` methods. The input to a text transformer should be a pandas `DataFrame` consisting of one or more textual columns. Example usage:

```python
from TextTransformers import LDATransformer

transformer = LDATransformer(num_topics=20, tfidf=False)
dt_transformed = transformer.fit_transform(dt_text)

```

Four transformers are implemented: `LDATransformer`, `PVTransformer`, `BoNGTransformer`, and `NBLogCountRatioTransformer`.
    

## Predictive monitoring

