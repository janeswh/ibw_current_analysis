"""Script to get a total number of IPSCs identified using the MOD method"""

import os
import pandas as pd
from collections import defaultdict

from regex import F
from run_single_test import run_both_conditions
from aggregate_stats import *
from file_settings import FileSettings

from single_test import JaneCell

from plotting import *

# from ppt_plotting import *
import pdb

from run_datasets import get_datasets, get_data_info, run_dataset_analysis
from run_single_test import BothConditions
from single_test import JaneCell


def tally_single_file(dataset, csvfile, file_name):
    file = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/data",
        dataset,
        file_name,
    )

    # gets sweep info for all cells
    sweep_info = pd.read_csv(csvfile, index_col=0)

    # 0 initializes JaneCell class
    condition_sweeps = JaneCell(dataset, sweep_info, file, file_name)

    # 1 checks whether cell has a data_notes response before proceeding
    condition_sweeps.check_response()

    # 2 imports event timings from MOD and calculate event kinetics
    condition_sweeps.get_mod_events()

    return condition_sweeps


def tally_dataset(dataset):
    dataset_data_folder = os.path.join(FileSettings.DATA_FOLDER, dataset)

    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, dataset, csvfile_name)

    ibwfile_list = []

    # only adds light file
    for file in os.listdir(dataset_data_folder):
        if file.endswith("light100.ibw"):
            ibwfile_list.append(file)

    dataset_events_tally = 0

    for file_count, ibwfile_name in enumerate(ibwfile_list):
        cell_name = ibwfile_name.split("_light")[0]
        # run_both_conditions(dataset, csvfile, cell_name)
        cell = BothConditions(dataset, csvfile, cell_name)

        file_names = [
            f"{cell.cell_name}_light100.ibw",
            f"{cell.cell_name}_spontaneous.ibw",
        ]

        cell.light_sweeps = tally_single_file(
            cell.dataset, cell.csvfile, file_names[0]
        )
        cell.spon_sweeps = tally_single_file(
            cell.dataset, cell.csvfile, file_names[1]
        )

        cell_events_tally = len(cell.light_sweeps.mod_events_df) + len(
            cell.spon_sweeps.mod_events_df
        )
        dataset_events_tally = dataset_events_tally + cell_events_tally

        print(
            f"Analysis for {cell_name} done, "
            f"{file_count+1}/{len(ibwfile_list)} cells"
        )

    return dataset_events_tally


def main():
    dataset_list, all_patched = get_data_info()
    print(f"Total # of cells patched across all datasets: {all_patched}")
    dataset_list.sort(reverse=True)  # put p2 first

    all_tally = 0
    for dataset in dataset_list:
        dataset_tally = tally_dataset(dataset)
        all_tally = all_tally + dataset_tally

    pdb.set_trace()


if __name__ == "__main__":
    main()
