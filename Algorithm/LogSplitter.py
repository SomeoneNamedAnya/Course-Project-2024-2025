from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

class LogSplitter:
    '''
        The number of splitting part
        method = kmeans | dbscan
    '''
    def __init__(self, k, method="kmeans"):
        self.k = k
        self.words = None
        self.case_label = None
        self.method = method
    
    '''
        Convert activity from event log to vector based on traces
    '''
    def fit(traces, vector_size=10, sg=0, seed=12345):
        model = Word2Vec(traces, vector_size=vector_size, min_count=1, sg=sg)
        self.words = {word: model.wv[word] for word in model.wv.index_to_key}

    '''
        Splitting event log based on event log by using different clustering algorythm. 
    '''
    def transform(log, parts=2, trace_parts=0.1):
        # log["id"]
        # log["traces"]
        # log["resourse"]
        encode_activity = list()
        
        len_part = max(log["traces"].apply(lambda x : len(x)).median() * trace_parts, 1)
        id_ref = dict()

        for id, trace in zip(log["id"], log["traces"]):
            encode_activity.append(list())
            start = len(encode_activity) - 1

            temp_count = 0

            if len(trace) < len_part:
                while temp_count < len_part:
                    activity = trace[temp_count % len(trace)]
                    if activity not in self.words.keys():
                        raise Exception("Doesn't exist activity \"" + activity + "\" in training data")
                    
                    encode_activity[-1].append(self.words[activity])
                    temp_count += 1
                id_ref[id] = (start, start + 1)
                continue

            for num in range(len(trace)):
                activity = trace[num]
                if activity not in self.words.keys():
                    raise Exception("Doesn't exist activity \"" + activity + "\" in training data")
                
                encode_activity[-1].append(self.words[activity])
                temp_count += 1

                if temp_count == len_part:
                    temp_count = 0
                    if len(trace) - (num + 1) < len_part:
                        num = len(trace) - len_part - 1
                    encode_activity.append(list())

            end = len(encode_activity)

            id_ref[id] = (start, end)
        
        
        
        labels = list()
        if self.method == "kmeans":
            labels = self.alg_KMeans(encode_activity)
        elif self.method == "dbscan":
            labels = self.alg_dbscan(encode_activity)
        
        self.case_label = list()

        for id in id_ref.keys:

            start, end = id_ref[id]
            frequency = np.array([0 for _ in range(parts)])
            for i in range(start, end):
                frequency[labels[i]] += 1
            self.case_label.append(np.argmax(frequency))
        
        return self.case_label
    
    '''
        Kmeans algorythm from sklearn
    '''
    def alg_KMeans(encode_activity):

        model = KMeans(n_clusters = self.k, n_init = 20, max_iter=500, random_state = 17)

        return model.fit_predict(encode_activity)
    
    def alg_dbscan(encode_activity):

        model = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
    
        return model.fit_predict(encode_activity)
    




        


        

        



            




        
    