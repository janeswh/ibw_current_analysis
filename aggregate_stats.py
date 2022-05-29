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


class CellTypeSummary(object):
    def __init__(self, dataset, cell_type):
        self.dataset = dataset
        self.cell_type = cell_type
        self.cell_type_stats_folder = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, cell_type
        )
        self.cell_type_figures_folder = os.path.join(
            FileSettings.FIGURES_FOLDER, dataset, cell_type
        )

        self.threshold = None
        self.stim_conditions = None
        self.cell_counts = None

        self.raw_responses_df = None
        self.responding_cells = None
        self.responding_cells_num = None
        self.non_responding_cells = None
        self.nonresponding_cells_num = None
        self.mismatch_cells = None

        self.win_median_stats_df = None
        self.outsid_median_stats_df = None

        self.respon_avg_freq_stats = None
        self.mean_trace_stats = None

    def get_responding_cells(self):
        """
        Loops through each cell's responses.csv to check its "overall response"
        value to see whether it has a response.
        """
        responses_list = []

        for cell_name in os.listdir(os.path.join(self.cell_type_stats_folder)):
            for filename in os.listdir(
                os.path.join(self.cell_type_stats_folder, cell_name)
            ):
                if filename.endswith("responses.csv"):
                    responses_list.append(
                        os.path.join(
                            self.cell_type_stats_folder, cell_name, filename
                        )
                    )

        self.raw_responses_df = pd.concat(
            [
                pd.read_csv(file, header=0)  # cell_name is idx
                for file in responses_list
            ],
            axis=0,
        )

        self.responding_cells = list(
            self.raw_responses_df.loc[
                self.raw_responses_df["overall response"] == True
            ]["cell_name"]
        )
        self.non_responding_cells = list(
            self.raw_responses_df.loc[
                self.raw_responses_df["overall response"] == False
            ]["cell_name"]
        )

        self.responding_cells_num = len(self.responding_cells)
        self.nonresponding_cells_num = len(self.non_responding_cells)

    def get_response_mismatches(self):
        """
        Compares the response values from each file with those determined by
        eye in data_notes
        """

        self.mismatch_cells = list(
            self.raw_responses_df.loc[
                self.raw_responses_df["vs. eye mismatch"] == True
            ]["cell_name"]
        )

    def get_mean_trace_stats(self):
        """
        Reads in the mean trace stats for responding cells, light condition
        only
        """
        cell_list = self.responding_cells
        mean_trace_stats_files = []

        for cell_name in cell_list:
            for filename in os.listdir(
                os.path.join(self.cell_type_stats_folder, cell_name)
            ):
                if filename.endswith("mean_trace_stats.csv"):
                    mean_trace_stats_files.append(
                        os.path.join(
                            self.cell_type_stats_folder, cell_name, filename
                        )
                    )

        mean_trace_stats = pd.concat(
            [
                pd.read_csv(file, index_col=0, header=0)
                for file in mean_trace_stats_files
            ]
        )

        # only saves the stats for light conditions
        self.mean_trace_stats = mean_trace_stats.loc[
            mean_trace_stats["Condition"] == "Light"
        ]

    def get_event_median_stats(self):
        # reads in median event stats for within response window and outside
        win_file_paths = []
        outside_file_paths = []

        # this is all the cells
        # cell_list = os.listdir(os.path.join(self.cell_type_stats_folder))

        # this is responding cells only
        cell_list = self.responding_cells

        for cell_name in cell_list:
            for filename in os.listdir(
                os.path.join(self.cell_type_stats_folder, cell_name)
            ):
                if filename.endswith("median_window_event_stats.csv"):
                    win_file_paths.append(
                        os.path.join(
                            self.cell_type_stats_folder, cell_name, filename
                        )
                    )
                elif filename.endswith("median_outside_event_stats.csv"):
                    outside_file_paths.append(
                        os.path.join(
                            self.cell_type_stats_folder, cell_name, filename
                        )
                    )
        self.win_median_stats_df = pd.concat(
            [
                pd.read_csv(file, index_col=0, header=0)
                for file in win_file_paths
            ]
        )

        self.outside_median_stats_df = pd.concat(
            [
                pd.read_csv(file, index_col=0, header=0)
                for file in outside_file_paths
            ]
        )

    def get_avg_freq_stats(self):
        """
        Reads in the average frequency kinetics values
        """
        cell_list = self.responding_cells
        freq_stats_files = []

        for cell_name in cell_list:
            for filename in os.listdir(
                os.path.join(self.cell_type_stats_folder, cell_name)
            ):
                if filename.endswith("avg_freq_stats.csv"):
                    freq_stats_files.append(
                        os.path.join(
                            self.cell_type_stats_folder, cell_name, filename
                        )
                    )

        respon_avg_freq_stats = pd.concat(
            [
                pd.read_csv(file, index_col=0, header=0)
                for file in freq_stats_files
            ]
        )

        # only saves the stats for light conditions
        self.respon_avg_freq_stats = respon_avg_freq_stats.loc[
            respon_avg_freq_stats["Condition"] == "Light"
        ]

    def save_summary_avgs(self):
        csv_filename = "{}_{}_{}_thresh_summary_averages.csv".format(
            self.dataset, self.genotype, thresh_prefix(self.threshold)
        )
        path = os.path.join(self.genotype_stats_folder, csv_filename)
        self.selected_avgs.to_csv(path, float_format="%8.4f", index=False)


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


def get_cell_types(dataset):
    # listing the genotypes in each dataset
    stats_folder = os.path.join(FileSettings.TABLES_FOLDER, dataset)
    cell_types_list = [
        cell_type.name
        for cell_type in os.scandir(stats_folder)
        if cell_type.is_dir()
    ]

    return cell_types_list


# has threshold for loop leave
def get_cell_type_summary(dataset, cell_types_list):

    response_cells_list = pd.DataFrame()
    response_count_dict = defaultdict(dict)
    median_stats_dict = defaultdict(dict)
    freq_stats_dict = defaultdict(dict)
    mean_trace_stats_dict = defaultdict(dict)

    # genotypes_list = ["OMP"]
    for cell_type in cell_types_list:
        cell_type_summary = CellTypeSummary(dataset, cell_type)

        # gets responding cell counts
        cell_type_summary.get_responding_cells()
        response_cells_list = pd.concat(
            [response_cells_list, cell_type_summary.raw_responses_df]
        )

        response_count_dict[cell_type][
            "response"
        ] = cell_type_summary.responding_cells_num
        response_count_dict[cell_type][
            "no response"
        ] = cell_type_summary.nonresponding_cells_num

        # gets mean trace stats
        cell_type_summary.get_mean_trace_stats()
        mean_trace_stats_dict[cell_type][
            "mean trace stats"
        ] = cell_type_summary.mean_trace_stats

        # gets median event stats
        cell_type_summary.get_event_median_stats()
        median_stats_dict[cell_type][
            "response win"
        ] = cell_type_summary.win_median_stats_df
        median_stats_dict[cell_type][
            "outside win"
        ] = cell_type_summary.outside_median_stats_df

        # gets avg frequency stats
        cell_type_summary.get_avg_freq_stats()
        freq_stats_dict[cell_type][
            "avg freq stats"
        ] = cell_type_summary.respon_avg_freq_stats

    return (
        response_cells_list,
        mean_trace_stats_dict,
        response_count_dict,
        median_stats_dict,
        freq_stats_dict,
    )


def make_cell_type_summary_dfs(
    dataset, mean_trace_stats, counts, median_stats, freq_stats
):
    """
    Collects info from dicts and combines both cell types into one df per
    dataset
    """

    # makes cell counts df
    dataset_counts = pd.DataFrame()
    mean_trace_stats_df = pd.DataFrame()
    win_median_stats_df = pd.DataFrame()
    outside_median_stats_df = pd.DataFrame()
    freq_stats_df = pd.DataFrame()

    for cell_type in counts.keys():
        info_col = pd.DataFrame(
            {"timepoint": dataset, "cell type": cell_type}, index=[0]
        )

        # collects cell counts
        cell_type_counts = pd.DataFrame(counts[cell_type], index=[0])
        cell_type_counts = pd.concat([info_col, cell_type_counts], axis=1)
        dataset_counts = pd.concat([dataset_counts, cell_type_counts])

        # collects mean trace stats
        mean_trace = mean_trace_stats[cell_type]["mean trace stats"]
        mean_trace_stats_df = pd.concat([mean_trace_stats_df, mean_trace])

        # collects median stats
        win_stats = pd.DataFrame(median_stats[cell_type]["response win"])
        outside_stats = pd.DataFrame(median_stats[cell_type]["outside win"])

        win_median_stats_df = pd.concat([win_median_stats_df, win_stats])
        outside_median_stats_df = pd.concat(
            [outside_median_stats_df, outside_stats]
        )

        # collects freq stats
        avg_freq_stats = freq_stats[cell_type]["avg freq stats"]
        freq_stats_df = pd.concat([freq_stats_df, avg_freq_stats])

    return (
        dataset_counts,
        mean_trace_stats_df,
        win_median_stats_df,
        outside_median_stats_df,
        freq_stats_df,
    )


def save_dataset_stats(
    dataset,
    response_cells_list,
    mean_trace_stats,
    dataset_counts_df,
    win_stats,
    outside_stats,
    freq_stats,
):
    """
    Saves the response window and outside window median stats and avg freq
    stat dfs as csvs. Not saving dataset_counts because overall count is saved
    for all the datasets.
    """
    filenames = [
        f"{dataset}_response_cells_list.csv",
        f"{dataset}_mean_trace_stats.csv",
        f"{dataset}_response_counts.csv",
        f"{dataset}_window_median_stats.csv",
        f"{dataset}_outside_median_stats.csv",
        f"{dataset}_avg_frequency_stats.csv",
    ]
    dfs = [
        response_cells_list,
        mean_trace_stats,
        dataset_counts_df,
        win_stats,
        outside_stats,
        freq_stats,
    ]

    for count, df in enumerate(dfs):
        filename = filenames[count]
        path = os.path.join(FileSettings.TABLES_FOLDER, dataset, filename)
        df.to_csv(path, float_format="%8.4f")


def save_response_counts(df_counts):

    csv_filename = "response_counts.csv"
    path = os.path.join(FileSettings.TABLES_FOLDER, csv_filename)
    df_counts.to_csv(path, float_format="%8.4f")


def do_cell_counts(response_cell_counts, all_patched):
    """
    Collects all the response cell counts from all datasets and puts it into
    a df
    """

    response_counts_df = pd.DataFrame()

    for dataset in list(response_cell_counts.keys()):
        dataset_counts = pd.DataFrame()
        for cell_type in list(response_cell_counts[dataset].keys()):
            cell_type_col = pd.DataFrame({"cell type": cell_type}, index=[0])
            cell_type_counts = pd.DataFrame(
                response_cell_counts[dataset][cell_type], index=[0]
            )
            cell_type_counts = pd.concat(
                [cell_type_col, cell_type_counts], axis=1
            )
            dataset_counts = pd.concat([dataset_counts, cell_type_counts])

        dataset_col = pd.DataFrame({"timepoint": dataset}, index=[0])
        dataset_col = pd.concat([dataset_col, dataset_col])
        dataset_counts = pd.concat([dataset_col, dataset_counts], axis=1)

        response_counts_df = pd.concat([response_counts_df, dataset_counts])


def make_all_dataset_dfs(counts, median_stats, freq_stats):
    """
    Collects all the dict info and puts in dfs
    """
    response_counts_df = pd.DataFrame()
    freq_stats_df = pd.DataFrame()

    win_median_stats_df = pd.DataFrame()
    for dataset in counts.keys():
        for cell_type in counts[dataset].keys():
            cell_type_col = pd.DataFrame({"cell type": cell_type}, index=[0])
            pdb.set_trace()

