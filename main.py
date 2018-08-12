import os
import logging

# own imports
from data_import import Data
from genetic_MDPVRP import MDCPVRPSolution
from parameter_tuning import regression_tree_tuning
import openpyxl as xl
from individual import random_instance

# MAIN LOGGER (ROOT)
logging.getLogger().setLevel(logging.INFO)

SOLUTION_FILE = "results.xlsx"


def get_project_data(nr_data_files=None, file_numbers={"p": [], "pr": []}):
    """
    Retrieve and build all data objects used in this project

    Args:
        nr_data_files: Number of data files to be imported (standard is 35)

    Returns:

    """
    if nr_data_files is None:
        nr_data_files = 35

    data_objects = []
    cwd = os.getcwd()
    # get the first 23 files with name p<integer>
    if file_numbers["p"] is not None:
        if not file_numbers["p"]:
            file_nrs = range(1, min(nr_data_files + 1, 23 + 1))
        else:
            file_nrs = file_numbers["p"]

        for i in file_nrs:
            dat_path = cwd + "/data/p" + str(i).zfill(2) + ".txt"  # iterate through files
            data_object = Data(dat_path)

            bks_path = cwd + "/data/bks/p" + str(i).zfill(2) + ".res"  # iterate through files
            data_object.set_bks(bks_path)

            data_objects = data_objects + [data_object]

    if file_numbers["pr"] is not None:
        if not file_numbers["pr"]:
            file_nrs = min(nr_data_files - 23 + 1, 10 + 1)
        else:
            file_nrs = file_numbers["pr"]

        # get the last 10 files with name pr<integer>
        for i in file_nrs:
            path = cwd + "/data/pr" + str(i).zfill(2) + ".txt"  # iterate through files
            data_object = Data(path)

            bks_path = cwd + "/data/bks/pr" + str(i).zfill(2) + ".res"  # iterate through files
            data_object.set_bks(bks_path)

            data_objects = data_objects + [Data(path)]
    return data_objects


def main(rules=None):
    """
    Calculate and document the computational results

    """
    # file_numbers = {"p": [6, 8, 9, 10, 11], "pr":None}
    file_numbers = {"p": [6,10,12], "pr": [1,2,3,4]}
    data_files = get_project_data(file_numbers=file_numbers)
    for i, data in enumerate(data_files):
        for j in range(5):
            if rules is None:
                inst = MDCPVRPSolution(data)
            else:
                inst = MDCPVRPSolution(data, **rules)

            inst.solve()
            try:
                best_solution_value = inst.solution.length
            except Exception:
                best_solution_value = None
            bks = inst.vrp_data.bks
            info = ["p" + str(i+1).zfill(2), j, best_solution_value, bks]
            logging.info(best_solution_value)
            wb = xl.load_workbook(SOLUTION_FILE)
            sht = wb["results"]
            sht.append(info)
            wb.save(SOLUTION_FILE)


def parameter_tuning():
    cwd = os.getcwd()
    path = cwd + "/data/p06.txt"
    data = Data(path)
    inst = MDCPVRPSolution(data)
    base_rules = {"min_sub_pop_size": [10, 50],
                  "population_size": [10, 150],
                  "n_close_factor": [0.1, 1],
                  "target_proportion": [0.05, 0.7],
                  "nr_elite_individuals": [0.2, 1],
                  "iter_diversification": [0.1, 1],
                  "repair_probability": [0.1, 1],
                  "education_probability": [0.2, 1.0]}
    # select outlines for the parameters
    dtypes = {"min_sub_pop_size": "int",
              "population_size": "int",
              "n_close_factor": "float",
              "target_proportion": "float",
              "nr_elite_individuals": "float",
              "iter_diversification": "float",
              "repair_probability": "float",
              "education_probability": "float"}
    reduced_rules, x, y, reg_tree = regression_tree_tuning(inst, dtypes, **base_rules)
    print(reduced_rules)


if __name__ == "__main__":
    #parameter_tuning()

    # COMPUATIONAL RESULTS
    main()