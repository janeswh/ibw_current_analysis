import os
import pandas as pd
from collections import defaultdict
from run_single_test import run_both_conditions
from aggregate_stats import *
from file_settings import FileSettings

import pdb


def get_datasets():
    dataset_list = ["p2"]
    # dataset_list = [
    #     dataset
    #     for dataset in os.listdir(FileSettings.DATA_FOLDER)
    #     if dataset not in FileSettings.IGNORED
    # ]
    return dataset_list


def get_data_info():
    # gets the list of datasets from file directory
    dataset_list = get_datasets()
    # dataset_list = ["5dpi"]

    # runs stats analysis for each dataset

    recorded_counts = get_patched_counts(dataset_list)
    all_patched = make_patched_counts_df(recorded_counts)

    return dataset_list, all_patched


def run_dataset_analysis(dataset):
    dataset_data_folder = os.path.join(FileSettings.DATA_FOLDER, dataset)

    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, dataset, csvfile_name)

    ibwfile_list = []

    # only adds light file
    for file in os.listdir(dataset_data_folder):
        if file.endswith("light100.ibw"):
            ibwfile_list.append(file)

    for file_count, ibwfile_name in enumerate(ibwfile_list):
        cell_name = ibwfile_name.split("_light")[0]
        run_both_conditions(dataset, csvfile, cell_name)
        print(
            f"Analysis for {cell_name} done, "
            f"{file_count+1}/{len(ibwfile_list)} cells"
        )
    pdb.set_trace()


def main():
    dataset_list, all_patched = get_data_info()
    dataset_cell_counts = {}

    for dataset_count, dataset in enumerate(dataset_list):
        print("***Starting analysis for {} dataset.***".format(dataset))
        run_dataset_analysis(dataset)
        genotypes_list = get_genotypes(dataset)
        monosyn_cell_counts = get_genotype_summary(dataset, genotypes_list)

        dataset_cell_counts[dataset] = monosyn_cell_counts
        print(
            "***Analysis for {} dataset done, #{}/{} datasets.***".format(
                dataset, dataset_count + 1, len(dataset_list)
            )
        )
    # pdb.set_trace()
    do_cell_counts(dataset_cell_counts, all_patched)

    collect_selected_averages(dataset_cell_counts)


if __name__ == "__main__":

    main()
