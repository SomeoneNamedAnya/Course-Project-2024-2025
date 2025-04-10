from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.cluster import HDBSCAN
import numpy as np
import pandas as pd


class LogSplitter:
    """
        The number of splitting part
        method = kmeans | dbscan
    """

    def __init__(self, k, method="kmeans"):
        self.event_logs = None
        self.id_to_class = None
        self.vector_size = None
        self.name_traces = None
        self.name_id = None
        self.k = k if method == "kmeans" else 0
        self.words = None
        self.case_label = None
        self.method = method

    '''
        Convert activity from event log to vector based on traces
    '''

    def fit(self, traces, vector_size=10, sg=0):
        model = Word2Vec(traces, vector_size=vector_size, min_count=1, sg=sg)
        temp_name_word = list()
        self.vector_size = vector_size
        temp_vec_words = list()
        for word in model.wv.index_to_key:
            temp_name_word.append(word)
            temp_vec_words.append(model.wv[word])
        temp_vec_words = normalize(temp_vec_words, norm="l2")
        self.words = {temp_name_word[i]: temp_vec_words[i] for i in range(len(temp_vec_words))}


    '''
        Splitting event log based on event log by using different clustering algorythm. 
    '''

    def transform(self, log, name_id='id', name_traces='traces', trace_parts=0.2, choose_func=1):
        self.name_id = name_id
        self.name_traces = name_traces
        # log["id"]
        # log["traces"]
        # log["resourse"]
        encode_activity = list()
        
        if choose_func == 1:
            len_part = int(max(log[self.name_traces].apply(lambda x: len(x)).median() * trace_parts, 1))
        else:
            len_part = int(max(log[self.name_traces].apply(lambda x: len(x)).mean() * trace_parts, 1))

        self.len_part = len_part
        id_ref = dict()

        for id, trace in zip(log[self.name_id], log[self.name_traces]):
            encode_activity.append(np.zeros(self.vector_size))
            start = len(encode_activity) - 1

            temp_count = 0

            if len(trace) < len_part:
                while temp_count < len_part:
                    activity = trace[temp_count % len(trace)]
                    if activity not in self.words.keys():
                        raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                    encode_activity[-1] += self.words[activity]
                    temp_count += 1
                id_ref[id] = (start, start + 1)
                continue

            for num in range(len(trace)):
                activity = trace[num]
                if activity not in self.words.keys():
                    raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                encode_activity[-1] += self.words[activity]
                temp_count += 1

                if temp_count == len_part:
                    temp_count = 0
                    if num + 1 == len(trace):
                        break
                    elif len_part + num >= len(trace):
                        encode_activity.append(np.zeros(self.vector_size))
                        for temp_num in range(len(trace) - len_part, len(trace)):
                            activity = trace[temp_num]
                            if activity not in self.words.keys():
                                raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                            encode_activity[-1] += self.words[activity]
                        break

                    else:
                        encode_activity.append(np.zeros(self.vector_size))

            end = len(encode_activity)

            id_ref[id] = (start, end)

        labels = list()
        

        if self.method == "kmeans":
            labels = self.alg_kmeans(encode_activity)
        elif self.method == "dbscan":
            labels = self.alg_dbscan(encode_activity, eps)

        self.id_to_class = dict()

        self.class_to_is = dict()
        for i in range(self.k):
            self.class_to_is[i] = 0

        for id in id_ref.keys():

            start, end = id_ref[id]
            frequency = np.array([0 for _ in range(self.k)])
            for i in range(start, end):
                frequency[labels[i]] += 1

            self.k = max(self.k, np.argmax(frequency))
            self.id_to_class[id] = np.argmax(frequency)
            self.class_to_is[self.id_to_class[id]] += 1 

        return

    '''
        Kmeans algorythm from sklearn
    '''

    def alg_kmeans(self, encode_activity):
       # encode_activity = normalize(encode_activity, norm="l2")
        model = KMeans(n_clusters=self.k)

        return model.fit_predict(encode_activity)

    '''
        DBSCAN algorythm from sklearn
    '''

    def alg_dbscan(self, encode_activity, eps):

       # model = HDBSCAN(eps=0.1, min_samples=5, metric="cosine")
        model = HDBSCAN(min_cluster_size=5, metric="cosine")
        return model.fit_predict(encode_activity)

    '''
        Split event logs into sub logs
    '''

    def get_logs(self, event_log):
        self.event_logs = [list() for _ in range(self.k)]
        for i in range(event_log.shape[0]):
            class_id = self.id_to_class[int(event_log[self.name_id].loc[i])]
            self.event_logs[class_id].append(event_log.iloc[i].to_dict())

        return self.event_logs

    '''
        save as csv splitting event logs into sub logs
    '''

    def save_as_csv(self, event_log, path=""):

        self.get_logs(event_log)

        for i in range(len(self.event_logs)):
            pd.DataFrame(self.event_logs[i]).to_csv(path + str(i) + ".csv", index=False)



