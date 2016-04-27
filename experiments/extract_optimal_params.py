import pandas as pd
import numpy as np
import re
import os

files = os.listdir("cv_results/")

all_metrics = {}
for file in files:
    parts = file.split("_")
    if parts[0] == "bong":
        m = re.match("selected(\d+)", parts[1])
        selected = m.group(1)
        
        if parts[2] == "tfidf":
            tfidf = "tfidf"
            nextidx = 3
        else:
            tfidf = "no-tfidf"
            nextidx = 2
            
        m = re.match("ngram(\d+)", parts[nextidx])
        ngram = m.group(1)
        
        cls = parts[nextidx+1]
        
        m = re.match("part(\d+)", parts[nextidx+2])
        part = m.group(1)
        
        metrics = pd.read_csv("cv_results/%s"%file, sep=";")
        metrics["selected"] = selected
        metrics["tfidf"] = tfidf
        metrics["ngram"] = ngram
        
    elif parts[0] == "nb":
        m = re.match("selected(\d+)", parts[1])
        selected = m.group(1)
        
        m = re.match("alpha(.*)", parts[2])
        alpha = m.group(1)
            
        m = re.match("ngram(\d+)", parts[3])
        ngram = m.group(1)
        
        cls = parts[4]
        
        m = re.match("part(\d+)", parts[5])
        part = m.group(1)
        
        metrics = pd.read_csv("cv_results/%s"%file, sep=";")
        metrics["selected"] = selected
        metrics["alpha"] = alpha
        metrics["ngram"] = ngram
        
    elif parts[0] == "lda":
        m = re.match("k(\d+)", parts[1])
        k = m.group(1)
        
        if parts[2] == "tfidf":
            tfidf = "tfidf"
            nextidx = 3
        else:
            tfidf = "no-tfidf"
            nextidx = 2
            
        cls = parts[nextidx]
        
        m = re.match("part(\d+)", parts[nextidx+1])
        part = m.group(1)
        
        metrics = pd.read_csv("cv_results/%s"%file, sep=";")
        metrics["k"] = k
        metrics["tfidf"] = tfidf
        
    elif parts[0] == "pv":
        m = re.match("size(\d+)", parts[1])
        size = m.group(1)
        
        m = re.match("window(\d+)", parts[2])
        window = m.group(1)
            
        cls = parts[3]
        
        m = re.match("part(\d+)", parts[4])
        part = m.group(1)
        
        metrics = pd.read_csv("cv_results/%s"%file, sep=";")
        metrics["size"] = size
        metrics["window"] = window
    else:
        continue

    metrics["part"] = part
    metrics["cls"] = cls
    if parts[0] not in all_metrics:
        all_metrics[parts[0]] = metrics.copy()
    else:
        all_metrics[parts[0]] = pd.concat([all_metrics[parts[0]], metrics], ignore_index=True)
        
optimal_params = {}
for k, v in all_metrics.items():
    grouped = v.groupby([col for col in v.columns if col not in ["part", "value"]])
    means = pd.DataFrame(grouped["value"].mean()).reset_index()
    fscores = means[means["metric"] == "fscore"].reset_index()
    fscores_max = fscores.loc[fscores.groupby(["confidence", "cls"])['value'].idxmax()]
    optimal_params[k] = fscores_max
    
for k, v in optimal_params.items():
    v.to_csv("cv_results/optimal_params_%s"%k, sep=";", index=False)