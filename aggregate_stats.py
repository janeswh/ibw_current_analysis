import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import sem
from collections import defaultdict
import plotly.io as pio
from file_settings import FileSettings
from plotting import *

pio.renderers.default = "browser"
import pdb
import glob


class GenotypeSummary(object):
    def __init__(self, dataset, genotype):
        self.dataset = dataset
        self.genotype = genotype
        self.genotype_stats_folder = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, genotype
        )
        self.genotype_figures_folder = os.path.join(
            FileSettings.FIGURES_FOLDER, dataset, genotype
        )
        self.concat_df = self.get_summary_stats()

        self.threshold = None
        self.stim_conditions = None
        self.cell_counts = None

    def get_summary_stats(self):
        # reads in all the summary stats values
        file_paths = []
        for filename in os.listdir(os.path.join(self.genotype_stats_folder)):
            if filename.endswith("response_stats.csv"):
                file_paths.append(
                    os.path.join(self.genotype_stats_folder, filename)
                )
        concat_df = pd.concat(
            [
                pd.read_csv(file, index_col=None, header=0)
                for file in file_paths
            ]
        )

        return concat_df

    def set_latency_threshold(self, threshold):
        # drops values from concat_df with mean onset latencies above threshold

        if threshold is None or threshold is "nothresh":
            self.thresh_concat = self.concat_df.copy()
        else:
            self.thresh_concat = self.concat_df[
                self.concat_df["Mean Onset Latency (ms)"] < threshold
            ]

        self.threshold = threshold

    def calc_summary_avgs(self):
        # this calculates stats grouped by stim conditions
        self.mean_df = self.thresh_concat.groupby(
            ["Light Intensity", "Light Duration"]
        ).mean()
        self.sem_df = self.thresh_concat.groupby(
            ["Light Intensity", "Light Duration"]
        ).sem()

    def get_summary_avgs(self):
        # gets avg values for the selected stim condition
        selected_intensity, selected_duration = FileSettings.SELECTED_CONDITION

        self.selected_avgs = self.thresh_concat[
            (self.thresh_concat["Light Intensity"] == selected_intensity)
            & (self.thresh_concat["Light Duration"] == selected_duration)
        ].copy()

        return self.selected_avgs

    def save_summary_avgs(self):
        csv_filename = "{}_{}_{}_thresh_summary_averages.csv".format(
            self.dataset, self.genotype, thresh_prefix(self.threshold)
        )
        path = os.path.join(self.genotype_stats_folder, csv_filename)
        self.selected_avgs.to_csv(path, float_format="%8.4f", index=False)

    def count_monosyn_cells(self):
        # count the number of cells with monosynaptic response used for each
        # stimulus condition after thresholding with mean onset latency

        # this creates a list of unique stim condition combinations and allows iteration
        self.stim_conditions = (
            self.thresh_concat[["Light Intensity", "Light Duration"]]
            .drop_duplicates(ignore_index=True)
            .copy()
        )

        conditions_tuples = list(
            zip(
                self.stim_conditions["Light Intensity"],
                self.stim_conditions["Light Duration"],
            )
        )

        cell_counts = pd.DataFrame()
        counts_list = []
        # this allows display of individual cell values matching each stim condition
        for condition in conditions_tuples:
            indiv_cells_df = self.thresh_concat[
                (self.thresh_concat["Light Intensity"] == condition[0])
                & (self.thresh_concat["Light Duration"] == condition[1])
            ].copy()
            num_cells = len(indiv_cells_df)

            counts_list.append(num_cells)

        cell_counts = pd.DataFrame(
            {(self.dataset + "/" + self.genotype): counts_list}
        )

        cell_counts.index = pd.MultiIndex.from_tuples(conditions_tuples)

        return cell_counts


def get_patched_counts(dataset_list):
    """
    Gets the number of files (i.e. # of cells patched) in all datasets
    using data_notes csv files.
    """

    recorded_counts = defaultdict(lambda: defaultdict(dict))

    for dataset in dataset_list:
        csvfile_name = "{}_data_notes.csv".format(dataset)
        csvfile = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, csvfile_name
        )
        cell_list = pd.read_csv(csvfile, index_col=0)

        # drops spontaneous files to get cell count
        cell_list = cell_list[cell_list["Response"].notna()]
        cell_types = cell_list["Cell Type"].unique()

        for cell_type in cell_types:
            cell_type_count = len(
                cell_list.loc[cell_list["Cell Type"] == cell_type]
            )
            cell_type_count = pd.DataFrame(
                {f"{dataset} / {cell_type}": cell_type_count}, index=[0]
            )
            recorded_counts[dataset][cell_type] = cell_type_count

    return recorded_counts


def make_patched_counts_df(recorded_counts):
    all_patched = pd.DataFrame()

    for dataset in recorded_counts.keys():
        # use the cell_types found in dict
        for cell_type in recorded_counts[dataset].keys():
            all_patched = pd.concat(
                [all_patched, recorded_counts[dataset][cell_type]], axis=1,
            )
            all_patched.fillna(0, inplace=True)
            all_patched = all_patched.astype(int)

    return all_patched


def get_genotypes(dataset):
    # listing the genotypes in each dataset
    stats_folder = os.path.join(FileSettings.TABLES_FOLDER, dataset)
    genotypes_list = [
        genotype.name
        for genotype in os.scandir(stats_folder)
        if genotype.is_dir()
    ]

    if len(genotypes_list) == 0:
        print(
            "{} dataset has no cells with responses, no summary stats calculated.".format(
                dataset
            )
        )

    return genotypes_list


# has threshold for loop leave
def get_genotype_summary(dataset, genotypes_list):
    monosyn_count_dict = defaultdict(dict)
    # genotypes_list = ["OMP"]
    for genotype in genotypes_list:
        genotype_summary = GenotypeSummary(dataset, genotype)
        genotype_summary.get_summary_stats()

        # counts_dict = {}  # one cell_counts df for each threshold
        # for threshold in [None, 2]:
        for threshold in FileSettings.THRESHOLD_LIST:
            if threshold is None:
                threshold = "nothresh"
            genotype_summary.set_latency_threshold(threshold)

            # this creates empty template cell_counts df for when no cells
            # exist
            if len(genotype_summary.thresh_concat) == 0:
                monosyn_cell_counts = monosyn_count_dict[genotype][
                    "nothresh"
                ].copy()
                monosyn_cell_counts.iloc[:, 0] = 0
            else:
                monosyn_cell_counts = genotype_summary.count_monosyn_cells()

                averages_fig = plot_averages(
                    genotype_summary.dataset,
                    genotype_summary.genotype,
                    threshold,
                    genotype_summary.thresh_concat,
                )

                save_summary_stats_fig(
                    genotype_summary.genotype,
                    threshold,
                    genotype_summary.genotype_figures_folder,
                    averages_fig,
                )

                genotype_summary.get_summary_avgs()

                # genotype_summary.calc_summary_avgs()
                genotype_summary.save_summary_avgs()

            monosyn_count_dict[genotype][threshold] = monosyn_cell_counts
            # print("{} threshold finished".format(threshold))

        # print("{} threshold finished".format(threshold))

    return monosyn_count_dict


def collect_selected_averages(counts_dict):
    all_selected_averages = {}
    for threshold in FileSettings.THRESHOLD_LIST:
        if threshold is None:
            threshold = "nothresh"
        file_paths = []
        for dataset in counts_dict.keys():
            # use the genotypes found in dict
            for genotype in counts_dict[dataset].keys():
                # pdb.set_trace()
                for filename in os.listdir(
                    os.path.join(FileSettings.TABLES_FOLDER, dataset, genotype)
                ):
                    if filename.endswith(
                        "{}_thresh_summary_averages.csv".format(
                            thresh_prefix(threshold)
                        )
                    ):
                        file_paths.append(
                            os.path.join(
                                FileSettings.TABLES_FOLDER,
                                dataset,
                                genotype,
                                filename,
                            )
                        )

        concat_avgs = pd.concat(
            [
                pd.read_csv(file, index_col=None, header=0)
                for file in file_paths
            ]
        )
        all_selected_averages[threshold] = concat_avgs

        # new things
        selected_summary_fig = plot_selected_averages(threshold, concat_avgs)
        save_selected_summary_fig(threshold, selected_summary_fig)

    return all_selected_averages


def get_monosyn_cell_counts(threshold, monosyn_counts_dict):
    """
    Takes the monosynaptic cell counts from all genotypes/datasets and 
    compiles into one df.
    """

    all_counts = pd.DataFrame()
    for dataset in monosyn_counts_dict.keys():
        # use the genotypes found in dict
        for genotype in monosyn_counts_dict[dataset].keys():
            all_counts = pd.concat(
                [
                    all_counts,
                    monosyn_counts_dict[dataset][genotype][threshold],
                ],
                axis=1,
            )

    all_counts.fillna(0, inplace=True)
    all_counts = all_counts.astype(int)

    save_cell_counts(all_counts, threshold)


def save_cell_counts(all_counts, threshold):
    """
    Takes the monosyn cell counts from specified threshold and saves as csv.
    """
    csv_filename = "{}_thresh_cell_counts.csv".format(thresh_prefix(threshold))
    path = os.path.join(FileSettings.TABLES_FOLDER, csv_filename)
    all_counts.to_csv(path, float_format="%8.4f")


def get_response_counts(threshold, all_counts, response_counts):
    """
    For each threshold, calculates:
    - % of cells that had a response / total # cells recorded in a dataset
    - % of cells that had a response / total # of cells that had a response 
    """
    csv_filename = "{}_thresh_cell_counts.csv".format(thresh_prefix(threshold))
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, csv_filename)
    counts_df = pd.read_csv(csvfile, index_col=[0, 1])
    response_counts = pd.DataFrame(
        counts_df.loc[FileSettings.SELECTED_CONDITION]
    ).T
    response_counts.index = [0]  # reset index for division
    no_response_counts = all_counts - response_counts
    response_dict = {}

    response_dict["response"] = response_counts
    response_dict["no response"] = no_response_counts

    return response_dict


def get_response_counts_df(threshold, response_counts_dict):
    """
    Puts all the counts for total recorded cells and monosynaptic responding
    cells for each latency into one df, then export as csv.
    """

    for response in response_counts_dict.keys():
        info = pd.DataFrame(
            {
                "Onset Latency Threshold (ms)": threshold,
                "Count Type": response,
            },
            index=[0],
        )
        temp_df = pd.concat([info, response_counts_dict[response]], axis=1)

    return temp_df


def save_response_counts(df_counts):

    csv_filename = "monosynaptic_response_counts.csv"
    path = os.path.join(FileSettings.TABLES_FOLDER, csv_filename)
    df_counts.to_csv(path, float_format="%8.4f")


def save_response_proportions(all_counts):

    response_proportions = pd.DataFrame()

    all_patched = all_counts[
        (all_counts["Count Type"] == "All patched")
        & (all_counts["Onset Latency Threshold (ms)"] == "N/A")
    ].copy()
    # drop str columns for division
    all_patched.drop(
        ["Onset Latency Threshold (ms)", "Count Type"], axis=1, inplace=True
    )

    nothresh_row = all_counts[
        (all_counts["Count Type"] == "response")
        & (all_counts["Onset Latency Threshold (ms)"] == "nothresh")
    ].copy()
    # drop str columns for division
    nothresh_row.drop(
        ["Onset Latency Threshold (ms)", "Count Type"], axis=1, inplace=True
    )

    for threshold in all_counts["Onset Latency Threshold (ms)"].unique():

        if not threshold in ["nothresh", "N/A"]:
            response_row = all_counts[
                (all_counts["Count Type"] == "response")
                & (all_counts["Onset Latency Threshold (ms)"] == threshold)
            ]
            #
            temp_response = response_row.drop(
                ["Onset Latency Threshold (ms)", "Count Type"], axis=1
            ).copy()

            response_allpatched = temp_response / all_patched
            response_allpatched[
                "Count Type"
            ] = "% responses out of all patched"
            response_allpatched["Onset Latency Threshold (ms)"] = threshold

            response_nothresh = temp_response / nothresh_row
            response_nothresh[
                "Count Type"
            ] = "% responses out of nothresh responses"
            response_nothresh["Onset Latency Threshold (ms)"] = threshold

            temp_proportions = pd.concat(
                [response_allpatched, response_nothresh]
            )

            response_proportions = pd.concat(
                [response_proportions, temp_proportions]
            )

    response_proportions = response_proportions[
        [
            "Onset Latency Threshold (ms)",
            "Count Type",
            "non-injected/OMP",
            "non-injected/Gg8",
            "3dpi/Gg8",
            "5dpi/OMP",
            "5dpi/Gg8",
            "dox_3dpi/Gg8",
            "dox_4dpi/Gg8",
            "dox_5dpi/Gg8",
        ]
    ]

    csv_filename = "monosynaptic_response_proportions.csv"
    path = os.path.join(FileSettings.TABLES_FOLDER, csv_filename)
    response_proportions.to_csv(path, float_format="%8.4f")


def do_cell_counts(monosyn_cell_counts, all_patched):
    """
    All the functions for counting cells with monosynaptic responses
    """
    # counts how many cells had monosynaptic responses after thresholding
    # with onset latency

    all_patched.drop("dox_3dpi_ctrl/Gg8", axis=1, inplace=True)

    response_counts_dict = defaultdict(dict)

    all_counts = pd.DataFrame()

    all_patched_copy = all_patched.copy()
    buffer_info = pd.DataFrame(
        {"Onset Latency Threshold (ms)": "N/A", "Count Type": "All patched",},
        index=[0],
    )
    all_patched_copy = pd.concat([buffer_info, all_patched_copy], axis=1)

    for threshold in FileSettings.THRESHOLD_LIST:
        threshold = threshold if threshold else "nothresh"
        # threshold = "nothresh" if threshold is None else threshold

        get_monosyn_cell_counts(threshold, monosyn_cell_counts)
        response_dict = get_response_counts(
            threshold, all_patched, monosyn_cell_counts
        )
        response_counts_dict[threshold] = response_dict

        # saves the counts and the calculated proportions for all datasets
        temp_df = get_response_counts_df(
            threshold, response_counts_dict[threshold]
        )
        all_counts = pd.concat([all_counts, temp_df])

    all_counts = pd.concat([all_patched_copy, all_counts])

    all_counts = all_counts[
        [
            "Onset Latency Threshold (ms)",
            "Count Type",
            "non-injected/OMP",
            "non-injected/Gg8",
            "3dpi/Gg8",
            "5dpi/OMP",
            "5dpi/Gg8",
            "dox_3dpi/Gg8",
            "dox_4dpi/Gg8",
            "dox_5dpi/Gg8",
        ]
    ]

    save_response_counts(all_counts)
    save_response_proportions(all_counts)

    # plots and save the cell counts
    response_counts_fig = plot_response_counts(response_counts_dict)
    save_response_counts_fig(response_counts_fig)


def thresh_prefix(threshold):
    if threshold == "nothresh" or threshold is None:
        prefix = "no_threshold"
    else:
        prefix = "{}ms".format(threshold)

    return prefix
