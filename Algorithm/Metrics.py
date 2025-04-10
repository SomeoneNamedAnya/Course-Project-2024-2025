import pm4py
import networkx as nx
import numpy as np
import pandas as pd


class Metric:

    def __init__(self):
        self.petri_net = list()
        return
    
    def make_petri_net(self, list_of_df):
        self.petri_net.clear()
        for df in list_of_df:
            self.petri_net.append(pm4py.discover_petri_net_inductive(
                df,
                activity_key='concept:name',
                case_id_key='case:concept:name',
                timestamp_key='time:timestamp',
                noise_threshold = 0.2
            ))
    
    def pt_cd(self, list_of_df):
        ans = list()
        weight = list()
        if len(self.petri_net) != 0:
            for df in self.petri_net:
                net, im, fm = df
                ans.append((len(net.arcs) / len(net.transitions) + len(net.arcs) / len(net.places)) / 2)
                weight.append(len(net.places))
        else:
            for df in list_of_df:
                net, im, fm = pm4py.discover_petri_net_inductive(
                    df,
                    activity_key='concept:name',
                    case_id_key='case:concept:name',
                    timestamp_key='time:timestamp',
                    noise_threshold = 0.2
                )
                ans.append((len(net.arcs) / len(net.transitions) + len(net.arcs) / len(net.places)) / 2)
                weight.append(len(net.places))
        ans = np.array(ans)
        weight = np.array(weight)
        weight_ans = ans * weight
        

        return (ans, ans.mean(), weight_ans.sum() / weight.sum())

    def e_cardoso(self, list_of_df):
        ans = list()
        weight = list()
        if len(self.petri_net) != 0:
            for df in self.petri_net:
                net, im, fm = df
                
                weight.append(len(net.places))
                cur_num = 0
                for place in net.places:
                    set_of_next = set()
                    for arc in net.arcs:
                        if arc.source != place:
                            continue
                        transition = arc.target
                        t_set = set()
                        for next_arc in net.arcs:
                            if next_arc.source == transition:
                                t_set.add(next_arc.target)
                        if len(t_set) != 0:
                            set_of_next.add(frozenset(t_set))
                    cur_num += len(set_of_next)
                ans.append(cur_num)

        else:

            for df in list_of_df:
                net, im, fm = pm4py.discover_petri_net_inductive(
                    df,
                    activity_key='concept:name',
                    case_id_key='case:concept:name',
                    timestamp_key='time:timestamp',
                    noise_threshold = 0.2
                )
                
                weight.append(len(net.places))
                cur_num = 0
                for place in net.places:
                    set_of_next = set()
                    for arc in net.arcs:
                        if arc.source != place:
                            continue
                        transition = arc.target
                        t_set = set()
                        for next_arc in net.arcs:
                            if next_arc.source == transition:
                                t_set.add(next_arc.target)
                        if len(t_set) != 0:
                            set_of_next.add(frozenset(t_set))
                    cur_num += len(set_of_next)
                ans.append(cur_num)

        ans = np.array(ans)
        weight = np.array(weight)
        weight_ans = ans * weight
    
        return (ans, ans.mean(), weight_ans.sum() / weight.sum())
    


