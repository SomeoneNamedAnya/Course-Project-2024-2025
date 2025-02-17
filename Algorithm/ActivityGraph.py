import pandas as pd
import numpy as np


class ActivityGraph:
    """
        Инициализация класса ActivityGraph
        Parameters
        ----------
        activities: int
                    количество активностей в журнале событий
        traces: list
                все трассировки в журнале событий
    """
    def __init__(self, activities: int, traces: list):
        self.__activity = activities
        self.__traces = traces.copy()
        self.__graph = self.__make_graph()

    """
        Строит граф активностей по переданным трассировкам из журнала событий
    """
    def __make_graph(self) -> list:
        temp_graph = [list() for i in range(self.__activity)]

        #event[0] = case_id, event[1] = activity
        # кодировать одну трассировку можно так (case_id, num_of_action, next_action)
        for trace in self.__traces:
            for event_id in range(len(trace)):
                case_id = trace[event_id][0]
                activity = trace[event_id][1]
                next_activity = trace[event_id + 1][1] if event_id + 1 != len(trace) else -1
                temp_graph[activity].append([case_id, event_id, next_activity])

        return temp_graph


if __name__ == '__main__':
    temp = ActivityGraph(5, [[[0, 1], [0, 2]]])
