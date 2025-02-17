
def common_part(trace, ind_t, sep, ind_s) -> int:
    ans = 0

    for i in range(ind_t, len(trace)):
        if trace[i][1] == sep[ind_s]:
            ans += 1
            ind_s += 1
            if ind_s > len(sep):
                break

    return ans


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
        temp_graph = [list() for _ in range(self.__activity)]

        #event[0] = case_id, event[1] = activity
        # кодировать одну трассировку можно так (case_id, num_of_action, next_action)
        for trace in self.__traces:
            for event_id in range(len(trace)):
                case_id = trace[event_id][0]
                activity = trace[event_id][1]
                next_activity = -1 if event_id + 1 == len(trace) else trace[event_id + 1][1]
                temp_graph[activity].append([case_id, event_id, next_activity])

        return temp_graph

    def method_h_0(self, power, width) -> (list, list):
        freq = [0 for _ in range(self.__activity)]
        path = [list() for _ in range(self.__activity)]

        for activity_id in range(self.__activity):

            path[activity_id]  = self.__dfs(activity_id,
                                            power,
                                            power,
                                            [int(i == activity_id) for i in range(self.__activity)])
            freq[activity_id] = len(path[activity_id])

        ind = freq.index(max(freq))
        print(path)

        return self.__divide(path[ind], width)


    def __divide(self, sep: list, width: int) -> (list, list):
        first = list()
        second = list()

        for case in range(len(self.__traces)):
            compare = 0
            for event in range(len(self.__traces[case])):
                activity = self.__traces[case][event][1]
                for i in range(len(sep)):
                    if activity == sep[i]:
                        compare = max(compare, common_part(self.__traces[case], activity, sep, i))
            if compare < width:
                first.append(self.__traces[case])
            else:
                second.append(self.__traces[case])

        return first, second

    def __dfs(self, ind: int, power_cur: int, power_next: int, level: list) -> list:
        cur_traces = 0
        cnt = dict()
        for event in range(len(self.__graph[ind])):
            case_id, event_id, next_activity = self.__graph[ind][event]
            if case_id not in cnt.keys():
                cnt[case_id] = list()
            cnt[case_id].append(next_activity)

        next_step = [0 for _ in range(self.__activity)]
        for case_id in cnt.keys():
            if len(cnt[case_id]) >= level[ind]:
                if cnt[case_id][level[ind] - 1] != -1:
                    next_step[cnt[case_id][level[ind] - 1]] += 1
                cur_traces += 1

        if cur_traces < power_cur:
            return []

        for_sort = [(next_step[i], i) for i in range(len(next_step))]
        for_sort.sort(reverse=True)

        path = list()
        for elem in for_sort:
            temp_count, next_activity = elem
            if temp_count < power_next:
                break
            level[next_activity] += 1
            temp_path = self.__dfs(next_activity, power_cur, power_next, level)
            if len(temp_path) > len(path):
                path = temp_path.copy()

            level[next_activity] -= 1

        return [ind] + path



if __name__ == '__main__':
    temp = ActivityGraph(6, [[[0, 1], [0, 2]],
                                            [[1, 1], [1, 2], [1, 3], [1, 4]],
                                            [[2, 5], [2, 1], [2, 2]],
                                            [[3, 4], [3, 5], [3, 2]]])
    print(temp.method_h_0(2, 2))
