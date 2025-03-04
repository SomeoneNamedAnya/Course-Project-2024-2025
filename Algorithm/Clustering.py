import numpy as np
import pandas as pd
import config
from sklearn.cluster import KMeans
import random
import preprocessing

class Clustering:

    def __init__(self, num_of_classes: int):
        self.event_chains = list()
        self.division = list()
        self.num_of_classes = num_of_classes

    def fit(self, event_chains):
        for i in range(len(event_chains)):
            #print(event_chains[i][1] )
            event_chains[i][1] = list(map(int, event_chains[i][1]))
            event_chains[i][0] = int(event_chains[i][0])
        self.event_chains = event_chains
        return self

    def split(self, type_of_convert: str,
              len_of_word: int, count_of_word=-1):

        pre_text = list()
        link = dict()

        for i in range(len(self.event_chains)):
            case_id, event = self.event_chains[i][0], self.event_chains[i][1]
            if len(event) < len_of_word:
                pre_text.append([case_id])
                pre_text[-1].extend(event)
                pre_text[-1].extend([0 for _ in range(len_of_word - len(event))])
                continue
            for j in range(0, len(event), len_of_word):
                pre_text.append([case_id])
                if len(event) <= len_of_word + j:
                    pre_text[-1].extend(event[len(event) - len_of_word:len(event)])
                else:
                    pre_text[-1].extend(event[j:min(len(event), j+len_of_word)])

        random.shuffle(pre_text)
        for ind in range(len(pre_text)):
            vec = pre_text[ind]
            if vec[0] not in link.keys():
                link[vec[0]] = list()
            link[vec[0]].append(ind)
        text = np.array([line[1:] for line in pre_text])
        #print(text)

        kmeans = KMeans(n_clusters=self.num_of_classes, random_state=0, n_init=10)
        kmeans.fit(text)
        #print(kmeans.labels_)
        array = kmeans.labels_

        for i in range(len(self.event_chains)):
            #print(array[link[i]])
            unique, count = np.unique(array[link[i]], return_counts=True)
            self.division.append(unique[np.argmax(count)])

        return self #.division

    def safe_as_scv(self, file):
        main_table = file

        link = [list() for _ in range(self.num_of_classes)]
        for i in range(main_table.shape[0]):

            link[self.division[int(main_table['case:concept:name_new'].loc[i])]].append(main_table.iloc[i].to_dict())

        for i in range(len(link)):
            pd.DataFrame(link[i]).to_csv("models/" + str(i) + ".csv", index=False)


if __name__ == '__main__':


    unique_case, unique_activity, df, X = preprocessing.get_traces(config.PATH_DATA_LOG_SECOND_CSV)

    print("acasc")
    Clustering(2).fit(X).split('s', 5).safe_as_scv(df)
