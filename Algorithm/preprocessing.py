import pm4py
import pandas as pd
import config

def convert():
    log_second = pm4py.read_xes(config.PATH_DATA_LOG_SECOND)
    table_second = pm4py.convert_to_dataframe(log_second)
    table_second.to_csv(config.PATH_DATA_LOG_SECOND_CSV, index=False)


def get_traces(name, case_id="case:concept:name", activity="concept:name"):
    df = pd.read_csv(name)
    if case_id not in df.columns or activity not in df.columns:
        raise KeyError("Неверно задан case_id или/и activity")

    new_case_id = case_id + "_new"
    new_activity = activity + "_new"

    code_case, unique_case = pd.factorize(df[case_id])
    code_activity, unique_activity = pd.factorize(df[activity])

    df[new_case_id] = code_case
    df[new_activity] = code_activity

    df_mod = df.groupby(df[new_case_id])[new_activity].agg(list).reset_index()
    df_list = df_mod.values.tolist()

    return unique_case, unique_activity, df, df_list



#if __name__ == '__main__':



