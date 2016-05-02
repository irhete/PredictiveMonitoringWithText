from PredictiveModel import PredictiveModel
import numpy as np
import os.path

class PredictiveMonitor():
    
    def __init__(self, event_nr_col, case_id_col, label_col, encoder_kwargs, cls_kwargs, transformer_kwargs,
                 pos_label=1, text_col=None, 
                 text_transformer_type=None, cls_method="rf"):
        
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.text_col = text_col
        self.pos_label = pos_label
        
        self.text_transformer_type = text_transformer_type
        self.cls_method = cls_method
        
        self.encoder_kwargs = encoder_kwargs
        self.transformer_kwargs = transformer_kwargs
        self.cls_kwargs = cls_kwargs
        
        self.models = {}
        self.predictions = {}
        self.evaluations = {}
    
    
    def train(self, dt_train, max_events=None):
        
        max_events = max(dt_train[self.event_nr_col]) if max_events==None else max_events
        for nr_events in range(1, max_events+1):
            
            pred_model = PredictiveModel(nr_events=nr_events, case_id_col=self.case_id_col, label_col=self.label_col, 
                                         text_col=self.text_col, text_transformer_type=self.text_transformer_type,
                                         cls_method=self.cls_method, encoder_kwargs=self.encoder_kwargs,
                                         transformer_kwargs=self.transformer_kwargs, cls_kwargs=self.cls_kwargs)

            pred_model.fit(dt_train)
            self.models[nr_events] = pred_model
    
    
    def test(self, dt_test, confidences=[0.6], two_sided=False, evaluate=True, output_filename=None, outfile_mode='w', performance_output_filename=None):
        
        for confidence in confidences:
            results = self._test_single_conf(dt_test, confidence, two_sided)
            self.predictions[confidence] = results
            
            if evaluate:
                evaluation = self._evaluate(dt_test, results, two_sided)
                self.evaluations[confidence] = evaluation
                
        if output_filename is not None:
            metric_names = list(self.evaluations[confidences[0]].keys())
            if not os.path.isfile(output_filename):
                outfile_mode = 'w'
            with open(output_filename, outfile_mode) as fout:
                if outfile_mode == 'w':
                    fout.write("confidence;value;metric\n")
                for confidence in confidences:
                    for k,v in self.evaluations[confidence].items():
                        fout.write("%s;%s;%s\n"%(confidence, v, k))
                        
        if performance_output_filename is not None:
             with open(performance_output_filename, 'w') as fout:
                    fout.write("nr_events;train_preproc_time;train_cls_time;test_preproc_time;test_time;nr_test_cases\n")
                    for nr_events, pred_model in self.models.items():
                        fout.write("%s;%s;%s;%s;%s;%s\n"%(nr_events, pred_model.preproc_time, pred_model.cls_time, pred_model.test_preproc_time, pred_model.test_time, pred_model.nr_test_cases))
                
    
    def _test_single_conf(self, dt_test, confidence, two_sided):

        results = []
        case_names_unprocessed = set(dt_test[self.case_id_col].unique())
        max_events = min(max(dt_test[self.event_nr_col]), max(self.models.keys()))

        nr_events = 1

        # monitor cases until confident prediction is made or the case ends
        while len(case_names_unprocessed) > 0 and nr_events <= max_events:
            
            # prepare test set
            dt_test = dt_test[dt_test[self.case_id_col].isin(case_names_unprocessed)]
            if len(dt_test[dt_test[self.event_nr_col] >= nr_events]) == 0: # all cases are shorter than nr_events
                break
            elif nr_events not in self.models:
                nr_events += 1
                continue
            
            # select relevant model
            pred_model = self.models[nr_events]
                
            # predict
            predictions_proba = pred_model.predict_proba(dt_test)

            # filter predictions with sufficient confidence
            for label_col_idx, label in enumerate(pred_model.cls.classes_):
                if label == self.pos_label or two_sided:
                    finished_idxs = np.where(predictions_proba[:,label_col_idx] >= confidence)
                    finished_cases = pred_model.test_case_names.iloc[finished_idxs]
                    for idx in finished_idxs[0]:
                        results.append({"case_name":pred_model.test_case_names.iloc[idx], 
                                        "prediction":label,
                                        "class":pred_model.test_y.iloc[idx],
                                        "nr_events":nr_events})
                        case_names_unprocessed = case_names_unprocessed.difference(set(finished_cases))
                
            nr_events += 1
        
        return(results)
        
        
    def _evaluate(self, dt_test, results, two_sided):
        case_lengths = dt_test[self.case_id_col].value_counts()
        dt_test = dt_test[dt_test[self.event_nr_col] == 1]
        N = len(dt_test)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        earliness = 0
        finished_case_names = [result["case_name"] for result in results]
        positives = sum(dt_test[self.label_col] == self.pos_label)
        negatives = sum(dt_test[self.label_col] != self.pos_label)

        
        for result in results:
            if result["prediction"] == self.pos_label and result["class"] == self.pos_label:
                tp += 1
            elif result["prediction"] == self.pos_label and result["class"] != self.pos_label:
                fp += 1
            elif result["prediction"] != self.pos_label and result["class"] != self.pos_label:
                tn += 1
            else:
                fn += 1
            earliness += 1.0 * result["nr_events"] / case_lengths[result["case_name"]]

        if not two_sided:
            dt_test = dt_test[~dt_test[self.case_id_col].isin(finished_case_names)] # predicted as negatives
            tn = sum(dt_test[self.label_col] != self.pos_label)
            fn = len(dt_test) - tn

        metrics = {}

        metrics["recall"] = 1.0 * tp / positives # alternative without failures: (tp+fn)
        metrics["accuracy"] = 1.0 * (tp+tn) / (tp+tn+fp+fn)
        if len(results) > 0:
            metrics["precision"] = 1.0 * tp / (tp+fp)
            metrics["earliness"] = earliness / len(results)
            metrics["fscore"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        else:
            metrics["precision"] = 0
            metrics["earliness"] = 0
            metrics["fscore"] = 0
        metrics["specificity"] = 1.0 * tn / negatives # alternative without failures: (fp+tn)
        metrics["tp"] = tp
        metrics["fn"] = fn
        metrics["fp"] = fp
        metrics["tn"] = tn
        metrics["failure_rate"] = 1 - 1.0 * len(results) / N
        
        return(metrics)