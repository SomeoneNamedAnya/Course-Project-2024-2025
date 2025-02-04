import pm4py
import pandas as pd
import config

if __name__ == "__main__":
    log_first = pm4py.read_xes(config.PATH_DATA_LOG_FIRST)
    table_first = pm4py.convert_to_dataframe(log_first)
    table_first.to_csv(config.PATH_DATA_LOG_FIRST_CSV, index=False)

    log_second = pm4py.read_xes(config.PATH_DATA_LOG_SECOND)
    table_second = pm4py.convert_to_dataframe(log_second)
    table_second.to_csv(config.PATH_DATA_LOG_SECOND_CSV, index=False)
