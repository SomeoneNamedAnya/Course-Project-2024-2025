from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd


class LogSplitter_2:
    """
        The number of splitting part
        method = kmeans | dbscan
    """

    def __init__(self, k, method="kmeans"):
        self.start = None
        self.max_len_class = None
        self.event_logs = None
        self.id_to_class = None
        self.name_traces = None
        self.name_id = None
        self.k = k if method == "kmeans" else 0
        self.words = None
        self.case_label = None
        self.method = method

    '''
        Convert activity from event log to vector based on traces
    '''

    def fit(self, traces, vector_size=10, sg=0, seed=12345):
        model = Word2Vec(traces, vector_size=1, min_count=1, sg=sg)
        self.words = {word: model.wv[word] for word in model.wv.index_to_key}

    '''
        Splitting event log based on event log by using different clustering algorythm. 
    '''

    def transform(self, log, name_id='id', name_traces='traces', parts=2, trace_parts=0.2):
        self.name_id = name_id
        self.name_traces = name_traces
        # log["id"]
        # log["traces"]
        # log["resourse"]
        encode_activity = list()

        len_part = int(max(log[self.name_traces].apply(lambda x: len(x)).median() * trace_parts, 1))
        id_ref = dict()

        for id, trace in zip(log[self.name_id], log[self.name_traces]):
            encode_activity.append(list())
            start = len(encode_activity) - 1

            temp_count = 0

            if len(trace) < len_part:
                while temp_count < len_part:
                    activity = trace[temp_count % len(trace)]
                    if activity not in self.words.keys():
                        raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                    encode_activity[-1].append(self.words[activity][0])
                    temp_count += 1
                id_ref[id] = (start, start + 1)
                continue

            for num in range(len(trace)):
                activity = trace[num]
                if activity not in self.words.keys():
                    raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                encode_activity[-1].append(self.words[activity][0])
                temp_count += 1

                if temp_count == len_part:
                    temp_count = 0
                    if num + 1 == len(trace):
                        break
                    elif len_part + num >= len(trace):
                        encode_activity.append(list())
                        for temp_num in range(len(trace) - len_part, len(trace)):
                            activity = trace[temp_num]
                            if activity not in self.words.keys():
                                raise Exception("Doesn't exist activity \"" + activity + "\" in training data")

                            encode_activity[-1].append(self.words[activity][0])
                        break

                    else:
                        encode_activity.append(list())

            end = len(encode_activity)

            id_ref[id] = (start, end)

        labels = list()

        if self.method == "kmeans":
            labels = self.alg_kmeans(encode_activity)
        elif self.method == "dbscan":
            labels = self.alg_dbscan(encode_activity)

        self.max_len_class = dict()

        self.start = labels[0]

        for id in id_ref.keys():

            start, end = id_ref[id]

            start_t = -1
            max_len = 0
            temp_len = 0
            temp_t = -1
            if start + 1 == end:
                self.max_len_class[id] = (len_part * 2, len_part * 2)
                continue

            for i in range(start + 1, end - 1):
                if labels[i] == 1 - self.start:
                    if temp_len == 0:
                        temp_t = i
                    temp_len += 1
                elif max_len < temp_len:
                    max_len = temp_len
                    start_t = temp_t
                    temp_t = -1
                    temp_len = 0
                else:
                    temp_t = -1
                    temp_len = 0
            if max_len < temp_len:
                max_len = temp_len
                start_t = temp_t

            self.max_len_class[id] = ((start_t - start) * len_part, (start_t - start) * len_part + max_len - 1)

            self.k = 2

        return

    '''
        Kmeans algorythm from sklearn
    '''

    def alg_kmeans(self, encode_activity):

        model = KMeans(n_clusters=self.k, n_init=20, max_iter=500, random_state=17)

        return model.fit_predict(encode_activity)

    '''
        DBSCAN algorythm from sklearn
    '''

    def alg_dbscan(self, encode_activity):

        model = DBSCAN(eps=0.5, min_samples=5, metric="cosine")

        return model.fit_predict(encode_activity)

    '''
        Split event logs into sub logs
    '''

    def get_logs(self, event_log):
        id_count = dict()

        self.event_logs = [list() for _ in range(self.k)]
        for i in range(event_log.shape[0]):
            pre_class_id = int(event_log[self.name_id].loc[i])
            if pre_class_id not in id_count.keys():
                id_count[pre_class_id] = 0
            id_count[pre_class_id] += 1
            start, end = self.max_len_class[pre_class_id]
            if id_count[pre_class_id] == start:
                temp_row = event_log.iloc[i].to_dict()
                temp_row["concept:name"] = 'start_new_sub_process'
                self.event_logs[1 - self.start].append(temp_row)
                temp_row["concept:name"] = 'new_sub_process'
                #self.event_logs[self.start].append(temp_row)
                self.event_logs[1 - self.start].append(event_log.iloc[i].to_dict())
            elif start < id_count[pre_class_id] < end:
                self.event_logs[1 - self.start].append(event_log.iloc[i].to_dict())
            elif id_count[pre_class_id] == end:
                temp_row = event_log.iloc[i].to_dict()
                temp_row["concept:name"] = 'end_new_sub_process'
                self.event_logs[1 - self.start].append(event_log.iloc[i].to_dict())
                self.event_logs[1 - self.start].append(temp_row)
            else:
                self.event_logs[self.start].append(event_log.iloc[i].to_dict())

        return self.event_logs

    '''
        save as csv splitting event logs into sub logs
    '''

    def save_as_csv(self, event_log, path=""):

        self.get_logs(event_log)

        for i in range(len(self.event_logs)):
            pd.DataFrame(self.event_logs[i]).to_csv(path + str(i) + ".csv", index=False)



