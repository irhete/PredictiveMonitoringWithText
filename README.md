# PredictiveMonitoringWithText
Scripts related to predictive business process monitoring framework with structured and unstructured (textual) data.


## Dependencies

* python 3.3
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [gensim](https://radimrehurek.com/gensim/) (for LDA and doc2vec models)
* [estnltk](https://github.com/estnltk/estnltk) (for lemmatization in Estonian language)



## Getting started


## Preprocessing

Before using the text models, textual data should be lemmatized. An example for Estonian language is given below.

    from estnltk import Text

    for idx, row in data.iterrows():
        text = Text(row["text_col"].lower())
        data.loc[idx]["lemmas_col"] = text.lemmas


## Text models


## Predictive monitoring

