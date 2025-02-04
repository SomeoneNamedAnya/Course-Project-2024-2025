import pm4py
DATA_PATH = "../Data/BPI_Challenge_2019/BPI_Challenge_2019.xes"

if __name__ == "__main__":
    log = pm4py.read_xes(DATA_PATH)
    process_model = pm4py.discover_bpmn_inductive(log)
    pm4py.view_bpmn(process_model)