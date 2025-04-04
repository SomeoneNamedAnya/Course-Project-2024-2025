import pm4py
import networkx as nx
import numpy as np
import pandas as pd


class Metric:

    def __init__(self):
        return
    
    def pt_cd(self, list_of_df):
        ans = list()
        weight = list()
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
    


