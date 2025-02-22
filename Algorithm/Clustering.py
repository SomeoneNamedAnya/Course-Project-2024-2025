import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random

class Clustering:
    '''
        event_chains - [[case_id, event_seq: []]

    '''
    def __init__(self, event_chains: list, num_of_classes: int):
        self.event_chains = list()
        self.division = list()
        self.num_of_classes = num_of_classes

    def fit(self, event_chains):
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
        main_table = pd.read_csv(file, sep=";")
        #print(main_table.columns)
        link = [list() for _ in range(self.num_of_classes)]
        for i in range(main_table.shape[0]):
            print(main_table.iloc[i].to_dict())
            link[self.division[main_table['case_id'].iloc[i]]].append(main_table.iloc[i].to_dict())

        for i in range(len(link)):
            pd.DataFrame(link[i]).to_csv("models/" + str(i) + ".csv", index=False)
if __name__ == '__main__':
    df = pd.read_csv("example.csv", sep=";")
    df2 = df.groupby("case_id").agg({"event_id":list})
    print(df2)
    conv = df2.to_dict()
    print(conv)
    X = list()
    for i in conv["event_id"].keys():
        X.append([i, conv["event_id"][i]])
    print(X)
   # X = list([[0, [1, 2, 3, 4]], [1, [1, 2, 5, 4]], [2, [1, 2, 1, 2, 3, 4]],
  #                [3, [5, 4]], [4, [1, 6, 7, 5, 4]], [5, [1, 5, 4]]])
    Clustering(X, 2).fit(X).split('s', 3).safe_as_scv("example.csv")

    # Обучаем модель
    #kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    #kmeans.fit(X)

    # Выводим результаты
    #print(kmeans.labels_)  # Метки кластеров
    #print(kmeans.cluster_centers_)  # Координаты центроидов
    #t = [[1000, 1, 2, 3],[-101000, 1, 5, 8],[-101000, 20, 5, 8], [-101000, 31, 5, 8]]

    #print(np.array([line[1:] for line in t]))