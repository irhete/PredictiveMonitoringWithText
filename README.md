This repository contains supplementary material for the article "[Predictive Business Process Monitoring with Structured and Unstructured Data](https://link.springer.com/chapter/10.1007/978-3-319-45348-4_23)" by [Irene Teinemaa](https://irhete.github.io/), [Marlon Dumas](http://kodu.ut.ee/~dumas/), [Fabrizio Maria Maggi](https://scholar.google.nl/citations?user=Jo9fNKEAAAAJ&hl=en&oi=sra), and [Chiara Di Francescomarino](https://shell-static.fbk.eu/people/dfmchiara/), which is published in the proceedings of the International Conference on Business Process Management 2016.

## Reference
If you use the code from this repository, please cite the original paper:
```
@inproceedings{teinemaa2016predictive,
  title={Predictive Business Process Monitoring with Structured and Unstructured Data},
  author={Teinemaa, Irene and Dumas, Marlon and Maggi, Fabrizio Maria and Di Francescomarino, Chiara},
  booktitle={International Conference on Business Process Management},
  pages={401--417},
  year={2016},
  organization={Springer}
}
```

## Dependencies

* python 3.5
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [gensim](https://radimrehurek.com/gensim/) (for LDA and doc2vec models)
* [estnltk](https://github.com/estnltk/estnltk) (for lemmatization in Estonian language)



## Preprocessing

Before using the text models, textual data should be lemmatized. The example below constructs a list of lemmatized documents (`docs_as_lemmas`), given a list of raw documents (`corpus`), using a lemmatizer for Estonian language.

```python
from estnltk import Text

docs_as_lemmas = []
for document in corpus:
    text = Text(document.lower())
    docs_as_lemmas.append(" ".join(text.lemmas))

```


## Sequence encoding

The `SequenceEncoder` enables encoding data as a complex sequence using index-based encoding. The input data should be in the following format:

    case_id;event_nr;class_label;dynamic_attr1;...;dynamic_attr_n;static_attr1;...;static_attr_h
    
In other words, each row in the input data should correspond to a given event (determined by `event_nr`) in a given case (determined by `case_id`). Each such event should be accompanied with a class label (`class_label`) that expresses the outcome of a case. Also, each event may carry an arbitrary number of static and dynamic data attributes. Both static and dynamic attributes may contain unstructured data, however, `SequenceEncoder` does not perform text processing by itself.

The output of sequence encoder is as follows:
    
    case_id;class_label;dynamic_attr1_event_1;...;dynamic_attr1_event_m;...;dynamic_attr_n_event_1;...;dynamic_attr_n_event_m;static_attr1;...;static_attr_h

When using `SequenceEncoder`, one should specify columns that represent the case id, event number, class label, dynamic attributes, and static attributes. Also, columns that should be interpreted as categorical values should be specified. Number of events that should be used for encoding the sequence (prefix length) is specified as the `nr_events` parameter. Cases that are shorter than `nr_events` are discarded.  Additionally, sequence encoder enables oversampling the dataset when `fit` is called (i.e. the training set), using `minority_label` as the class that should be oversampled. If `fillna=True`, all NA values are filled with zeros. Example usage of the sequence encoder is illustrated below.

```python
from SequenceEncoder import SequenceEncoder

encoder = SequenceEncoder(case_id_col="case_id", event_nr_col="event_nr", label_col="class_label", 
    static_cols=["static1", "static2", "static3"], dynamic_cols=["dynamic1", "dynamic2"], 
    cat_cols=["static2", "dynamic1"], nr_events=3, oversample_fit=True, minority_label="unsuccessful", 
    fillna=True, random_state=22)
train_encoded = encoder.fit_transform(train) # oversampled
test_encoded = encoder.transform(test)
```


## Text models

The text models are implemented as custom transformers, which include `fit`, `transform`, and `fit_transform` methods. 

Four transformers are implemented:
* `LDATransformer` - Latent Dirichlet Allocation topic modeling (utilizes Gensim's implementation).
* `PVTransformer` - Paragraph Vector (utilizes Gensim's implementation -- doc2vec)
* `BoNGTransformer` - bag-of-n-grams.
* `NBLogCountRatioTransformer` - bag-of-n-grams weighted with Naive Bayes log count ratios.

Example usage of the text transformers is shown below. `X` stands for a pandas `DataFrame` consisting of one or more textual columns, while `y` contains the target variable (class labels). Note that `X` should contain textual columns only.

```python
from TextTransformers import LDATransformer, PVTransformer, BoNGTransformer, NBLogCountRatioTransformer

lda_transformer = LDATransformer(num_topics=20, tfidf=False, passes=3, iterations=700, random_seed=22)
lda_transformer.fit(X)
lda_transformer.transform(X)

pv_transformer = PVTransformer(size=16, window=8, min_count=1, workers=1, alpha=0.025, dm=1, epochs=1, random_seed=22)
pv_transformer.fit(X)
pv_transformer.transform(X)

bong_transformer = BoNGTransformer(ngram_min=1, ngram_max=1, tfidf=False, nr_selected=100)
bong_transformer.fit(X, y)
bong_transformer.transform(X)

nb_transformer = NBLogCountRatioTransformer(ngram_min=1, ngram_max=1, alpha=1.0, nr_selected=100, pos_label="positive")
nb_transformer.fit(X, y)
nb_transformer.transform(X)

```


## Predictive model

The `PredictiveModel` class enables building a predictive model for a fixed prefix length, starting from raw data sets. The initializer expects as input the `text_transformer_type` (one of {`None`, "LDATransformer", "PVTransformer", "BoNGTransformer", "NBLogCountRatioTransformer"}) and classifier type `cls_method`, where "rf" stands for sklearn's `RandomForestClassifier` and "logit" stands for `LogisticRegression`. Furthermore, the prefix length should be predefined in `nr_events`, the names of relevant columns as `case_id_col`, `label_col`, `text_col`, and label of the positive class (`pos_label`). Additional arguments that should be forwarded to the `SequenceEncoder`, text transformer, and classifier should be given as `encoder_kwargs`, `transformer_kwargs`, and `cls_kwargs`, respectively (see sections above for details of these arguments).

Example usage:

```python
from PredictiveModel import PredictiveModel

encoder_kwargs = {"event_nr_col":event_nr_col, "static_cols":static_cols, "dynamic_cols":dynamic_cols,
                  "cat_cols":cat_cols,"oversample_fit":False, "minority_label":"unsuccessful",
                  "fillna":True, "random_state":22}
transformer_kwargs = {"ngram_max":ngram_max, "alpha":alpha, "nr_selected":nr_selected, 
                      "pos_label":pos_label}
cls_kwargs = {"n_estimators":500, "random_state":22}

pred_model = PredictiveModel(nr_events=nr_events, case_id_col=case_id_col, 
                             label_col=label_col, pos_label=pos_label, text_col=text_col, 
                             text_transformer_type="NBLogCountRatioTransformer", cls_method="rf",
                             encoder_kwargs=encoder_kwargs, transformer_kwargs=transformer_kwargs, 
                             cls_kwargs=cls_kwargs)

pred_model.fit(train)
predictions_proba = pred_model.predict_proba(test)
```

    

## Predictive monitoring

The `PredictiveMonitor` trains multiple `PredictiveModel`s (one for each possible prefix length) that consitute the offline component of the predictive monitoring framework. The arguments are the same as for `PredictiveModel`, with the exception of `event_nr_col` instead of `nr_events`. In the test phase, each case is monitored until a sufficient confidence level is achieved or the case ends. Possible arguments for testing function are a list of `confidences` to produce the results for, boolean `evaluate` if different metrics should be calculated, `output_filename` if the results should be written to an external file, and `performance_output_filename` if the calculation times should be written to an external file. 

Example usage:

```python
from PredictiveMonitor import PredictiveMonitor

encoder_kwargs = {"event_nr_col":event_nr_col, "static_cols":static_cols, "dynamic_cols":dynamic_cols,
                  "cat_cols":cat_cols, "oversample_fit":False, "minority_label":"unsuccessful", 
                  "fillna":True, "random_state":22}
transformer_kwargs = {"ngram_max":ngram_max, "alpha":alpha, "nr_selected":nr_selected, 
                  "pos_label":pos_label}
cls_kwargs = {"n_estimators":500, "random_state":22}

predictive_monitor = PredictiveMonitor(event_nr_col=event_nr_col, case_id_col=case_id_col,
                                      label_col=label_col, pos_label=pos_label, text_col=text_col,
                                      text_transformer_type="NBLogCountRatioTransformer", cls_method="rf",
                                      encoder_kwargs=encoder_kwargs, transformer_kwargs=transformer_kwargs, 
                                      cls_kwargs=cls_kwargs)

predictive_monitor.train(train)
predictive_monitor.test(test, confidences=[0.5, 0.75, 0.9], evaluate=True, output_filename="example_output.txt")
```

Real examples of predictive monitoring can be found in folder "experiments".
