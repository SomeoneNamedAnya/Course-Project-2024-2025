import pandas as pd

class EventLog:
    def __init__(self, cases):
        self.cases = cases
        self.graph = list(list(list()))
        self.graph.append([list(), list(), 'start'])

    def dfs(self, case, ind_trace, ind_graph) -> (int, int, bool):

        ans = (ind_trace, ind_graph, False)
        for out_ind in self.graph[ind_graph][1]:
            if self.graph[out_ind][2] == case[ind_trace].activity:
                ans = self.dfs(case, ind_trace + 1, out_ind)
                break
        if ind_trace + 1 == len(case):
            ans = (ind_trace, ind_graph, True)
        return ans

    def updateGraph(self, case, cur_ind_case, cur_ind_graph) -> int:
        for ind in range(cur_ind_case, len(case)):
            new_ind = len(self.graph)
            self.graph[cur_ind_graph][1].append(new_ind)
            self.graph.append([list(cur_ind_graph), list(), case.activity])
            cur_ind_graph = new_ind
        return cur_ind_graph

    def makeEndGraph(self, end_ind):
        self.graph.append([list(), list(), 'Finish'])
        id = len(self.graph) - 1
        for i in range(len(end_ind)):
            self.graph[id][0].append(end_ind[i])

        return True

    def makeGraph(self) -> bool:
        end_ind = list()
        for case in self.cases:
            flag = True
            for ind in self.graph[0][1]:
                cur_ind_case, cur_ind_graph, res = self.dfs(case, 0, 0)
                if not res:
                    end_ind.append(self.updateGraph(case, cur_ind_case, cur_ind_graph))
                else:
                    end_ind.append(cur_ind_graph)

        self.makeEndGraph(end_ind)

        return True

    def getTrace(self) -> list():
        return False

    def split(self) -> list():
        return False