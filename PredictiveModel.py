from SequenceEncoder import SequenceEncoder
from TextTransformers import LDATransformer, PVTransformer, BoNGTransformer, NBLogCountRatioTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

class PredictiveModel():

    def __init__(self, nr_events, case_id_col, label_col, encoder_kwargs, transformer_kwargs, cls_kwargs, text_col=None,
                 text_transformer_type=None, cls_method="rf"):
        
        self.text_col = text_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        
        self.encoder = SequenceEncoder(nr_events=nr_events, case_id_col=case_id_col, label_col=label_col, **encoder_kwargs)
        
        if text_transformer_type is None:
            self.transformer = None
        elif text_transformer_type == "LDATransformer":
            self.transformer = LDATransformer(**transformer_kwargs)
        elif text_transformer_type == "BoNGTransformer":
            self.transformer = BoNGTransformer(**transformer_kwargs)
        elif text_transformer_type == "NBLogCountRatioTransformer":
            self.transformer = NBLogCountRatioTransformer(**transformer_kwargs)
        elif text_transformer_type == "PVTransformer":
            self.transformer = PVTransformer(**transformer_kwargs)

        else:
            print("Transformer type not known")
        
        if cls_method == "logit":
            self.cls = LogisticRegression(**cls_kwargs) 
        elif cls_method == "rf":
            self.cls = RandomForestClassifier(**cls_kwargs)
        else:
            print("Classifier method not known")
        

    def fit(self, dt_train):
        
        train_encoded = self.encoder.fit_transform(dt_train)
        
        train_X = train_encoded.drop([self.case_id_col, self.label_col], axis=1)
        train_y = train_encoded[self.label_col]
        
        if self.transformer is not None:
            text_cols = [col for col in train_X.columns.values if col.startswith(self.text_col)]
            #train_text = self.transformer.fit_transform(train_X[text_cols], train_y)
            train_text = self.transformer.fit_transform(train_encoded[text_cols[len(text_cols)-1]], train_y)
            train_X = pd.concat([train_X.drop(text_cols, axis=1), train_text], axis=1)
        
        self.train_X = train_X
        self.cls.fit(train_X, train_y)

        
    def predict_proba(self, dt_test):
        test_encoded = self.encoder.transform(dt_test)
        
        test_X = test_encoded.drop([self.case_id_col, self.label_col], axis=1)
        
        if self.transformer is not None:
            text_cols = [col for col in test_X.columns.values if col.startswith(self.text_col)]
            #test_text = self.transformer.transform(test_encoded[text_cols])
            test_text = self.transformer.transform(test_encoded[text_cols[len(text_cols)-1]])
            test_X = pd.concat([test_X.drop(text_cols, axis=1), test_text], axis=1)
        
        self.test_case_names = test_encoded[self.case_id_col]
        self.test_X = test_X
        self.test_y = test_encoded[self.label_col]
        predictions_proba = self.cls.predict_proba(test_X)
        return predictions_proba