import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

class SequenceEncoder():
    
    def __init__(self, nr_events, event_nr_col, case_id_col, label_col, static_cols, dynamic_cols, 
                 cat_cols, oversample_fit=True, minority_label="positive", fillna=True, random_state=None):
        self.nr_events = nr_events
        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols
        self.cat_cols = cat_cols
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.oversample_fit = oversample_fit
        self.minority_label = minority_label
        self.random_state = random_state
        self.fillna = fillna
        
    def fit(self, X):
        return self
        
    def fit_transform(self, X):
        data = self._encode(X)
        if self.oversample_fit:
            data = self._oversample(data)
        return data
        
    def transform(self, X):
        data = self._encode(X)
        return data
        
    def _encode(self, X):
        # endoce static cols
        data_final = X[X[self.event_nr_col] == 1][self.static_cols]
        
        # encode dynamic cols
        for i in range(1, self.nr_events+1):
            data_selected = X[X[self.event_nr_col] == i][[self.case_id_col] + self.dynamic_cols]
            data_selected.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.dynamic_cols]
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="right")
        
        # make categorical
        dynamic_cat_cols = [col for col in self.cat_cols if col in self.dynamic_cols]
        static_cat_cols = [col for col in self.cat_cols if col in self.static_cols]
        catecorical_cols = ["%s_%s"%(col, i) for i in range(1, self.nr_events+1) for col in dynamic_cat_cols] + static_cat_cols
        cat_df = data_final[catecorical_cols]
        cat_dict = cat_df.T.to_dict().values()
        vectorizer = DV( sparse = False )
        vec_cat_dict = vectorizer.fit_transform(cat_dict)
        cat_data = pd.DataFrame(vec_cat_dict, columns=vectorizer.feature_names_)
        data_final = pd.concat([data_final.drop(catecorical_cols, axis=1), cat_data], axis=1)
        
        # fill NA
        if self.fillna:
            data_final = data_final.fillna(0)
            
        return data_final
    
    def _oversample(self, X):
        oversample_count = sum(X[self.label_col] != self.minority_label) - sum(X[self.label_col] == self.minority_label)

        if oversample_count > 0 and sum(X[self.label_col] == self.minority_label) > 0:
            oversampled_data = X[X[self.label_col]==self.minority_label].sample(oversample_count, replace=True, random_state=self.random_state)
            data = pd.concat([X, oversampled_data])

        return data
        