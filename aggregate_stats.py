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
import itertools

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
        self.all_mean_trace_stats = None

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
                self.raw_responses_df["datanotes eye response"] == True
            ]["cell_name"]
        )
        self.non_responding_cells = list(
            self.raw_responses_df.loc[
                self.raw_responses_df["datanotes eye response"] == False
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

    def get_mean_trace_stats(self, all_cells=False):
        """
        Reads in the mean trace stats for responding cells, light condition
        only
        """

        if all_cells == False:
            cell_list = self.responding_cells
        elif all_cells == True:
            cell_list = os.listdir(os.path.join(self.cell_type_stats_folder))

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

        if all_cells == False:
            # only saves the stats for light conditions
            self.mean_trace_stats = mean_trace_stats.loc[
                mean_trace_stats["Condition"] == "Light"
            ]
        elif all_cells == True:
            # only saves the stats for light conditions
            self.all_mean_trace_stats = mean_trace_stats.loc[
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

    all_mean_trace_stats_dict = defaultdict(dict)

    response_cells_list = pd.DataFrame()
    response_count_dict = defaultdict(dict)
    median_stats_dict = defaultdict(dict)
    freq_stats_dict = defaultdict(dict)
    mean_trace_stats_dict = defaultdict(dict)

    # nested_dict = lambda: defaultdict(nested_dict)
    # response_count_dict = nested_dict()

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

        # gets mean trace stats for responding cells only
        cell_type_summary.get_mean_trace_stats()
        mean_trace_stats_dict[cell_type][
            "mean trace stats"
        ] = cell_type_summary.mean_trace_stats

        # gets mean trace stats for all cells
        cell_type_summary.get_mean_trace_stats(all_cells=True)
        all_mean_trace_stats_dict[cell_type][
            "mean trace stats"
        ] = cell_type_summary.all_mean_trace_stats

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
        all_mean_trace_stats_dict,
        response_cells_list,
        mean_trace_stats_dict,
        response_count_dict,
        median_stats_dict,
        freq_stats_dict,
    )


def make_cell_type_summary_dfs(
    dataset,
    all_mean_trace_stats,
    cell_types_list,
    mean_trace_stats,
    counts,
    median_stats,
    freq_stats,
):
    """
    Collects info from dicts and combines both cell types into one df per
    dataset
    """
    # makes cell counts df
    all_mean_trace_stats_df = pd.DataFrame()

    dataset_counts = pd.DataFrame()
    mean_trace_stats_df = pd.DataFrame()
    win_median_stats_df = pd.DataFrame()
    outside_median_stats_df = pd.DataFrame()
    freq_stats_df = pd.DataFrame()

    for cell_type in cell_types_list:
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

        # collects mean trace stats for all cells
        all_mean_trace = all_mean_trace_stats[cell_type]["mean trace stats"]
        all_mean_trace_stats_df = pd.concat(
            [all_mean_trace_stats_df, all_mean_trace]
        )

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
        all_mean_trace_stats_df,
        dataset_counts,
        mean_trace_stats_df,
        win_median_stats_df,
        outside_median_stats_df,
        freq_stats_df,
    )


def save_dataset_stats(
    dataset,
    all_mean_trace_stats_df,
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
        f"{dataset}_all_mean_trace_stats.csv",
        f"{dataset}_response_cells_list.csv",
        f"{dataset}_mean_trace_stats.csv",
        f"{dataset}_response_counts.csv",
        f"{dataset}_window_median_stats.csv",
        f"{dataset}_outside_median_stats.csv",
        f"{dataset}_avg_frequency_stats.csv",
    ]
    dfs = [
        all_mean_trace_stats_df,
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
    path = os.path.join(
        FileSettings.TABLES_FOLDER, "datasets_summaries", csv_filename
    )
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


def get_ephys_sections_intensity():
    """
    Gets the fixed ephys section intensities to compare against response rates
    and mean peak freq of the recorded cells from each section.
    """
    sections_info = pd.read_csv(
        os.path.join(
            FileSettings.TABLES_FOLDER,
            "misc_csv_data",
            "ephys_GCL_intensity.csv",
        ),
        header=0,
    )

    # reads in cell response lists and avg freq stats
    all_responses = pd.DataFrame()
    all_freq_stats = pd.DataFrame()
    for timepoint in sections_info["timepoint"].unique():
        timepoint_responses = pd.read_csv(
            os.path.join(
                FileSettings.TABLES_FOLDER,
                timepoint,
                f"{timepoint}_response_cells_list.csv",
            ),
            header=0,
            index_col=0,
        )
        all_responses = pd.concat([all_responses, timepoint_responses])

        timepoint_freq = pd.read_csv(
            os.path.join(
                FileSettings.TABLES_FOLDER,
                timepoint,
                f"{timepoint}_avg_frequency_stats.csv",
            ),
            header=0,
            index_col=0,
        )
        all_freq_stats = pd.concat([all_freq_stats, timepoint_freq])

    sections_data = pd.DataFrame()

    for count, section in sections_info.iterrows():

        intensity = section["integrated density/area"]

        # gets response ratio
        timepoint = section["timepoint"]
        cell_names = section["cell names"].split(", ")
        cell_responses = all_responses[
            all_responses["cell_name"].isin(cell_names)
        ]

        response_ratio = sum(cell_responses["datanotes eye response"]) / len(
            cell_responses["datanotes eye response"]
        )

        # gets avg freq stats
        cell_freqs = all_freq_stats[
            all_freq_stats["Cell name"].isin(cell_names)
        ]

        peak_freqs = cell_freqs["Baseline-sub Peak Freq (Hz)"].tolist()
        peak_freq_mean = cell_freqs["Baseline-sub Peak Freq (Hz)"].mean()
        peak_freq_sem = cell_freqs["Baseline-sub Peak Freq (Hz)"].sem()

        section_list = pd.DataFrame(
            [
                timepoint,
                intensity,
                response_ratio,
                peak_freqs,
                peak_freq_mean,
                peak_freq_sem,
            ]
        ).T
        sections_data = pd.concat([sections_data, section_list])

    sections_data.columns = [
        "Dataset",
        "Integrated density/area",
        "Response %",
        "Peak Frequency (Hz)",
        "Mean Peak Frequency (Hz)",
        "Peak Frequency SEM",
    ]

    return sections_data


def get_slice_amps():
    slicesfile_name = f"slices_list.csv"
    slicesfile = os.path.join(
        FileSettings.TABLES_FOLDER, "misc_csv_data", slicesfile_name
    )
    slices_list = pd.read_csv(slicesfile)

    for timepoint in slices_list["Timepoint"].unique():
        timepoint_slices_list = slices_list.loc[
            slices_list["Timepoint"] == timepoint
        ]

        meantrace_file_name = f"{timepoint}_all_mean_trace_stats.csv"
        mean_trace_file = os.path.join(
            FileSettings.TABLES_FOLDER, timepoint, meantrace_file_name,
        )
        mean_trace_stats = pd.read_csv(mean_trace_file, index_col=0)

        # freq_file_name = f"{timepoint}_avg_frequency_stats.csv"
        # freq_file = os.path.join(
        #     FileSettings.TABLES_FOLDER, timepoint, freq_file_name,
        # )
        # freq_stats = pd.read_csv(freq_file, index_col=0)

        for cell in timepoint_slices_list["Cell name"].unique():
            cell_type = mean_trace_stats.loc[
                mean_trace_stats["Cell name"] == cell
            ]["Cell Type"][0]

            cell_mean_trace_peak = mean_trace_stats.loc[
                mean_trace_stats["Cell name"] == cell
            ]["Mean Trace Peak (pA)"][0]

            cell_log_peak = np.log(abs(cell_mean_trace_peak))

            slices_list.loc[
                slices_list["Cell name"] == cell, "Cell type"
            ] = cell_type
            slices_list.loc[
                slices_list["Cell name"] == cell, "Mean trace peak (pA)"
            ] = cell_mean_trace_peak
            slices_list.loc[
                slices_list["Cell name"] == cell, "Log mean trace peak"
            ] = cell_log_peak

    return slices_list


def get_slice_avg_amps(amps):
    """
    Gets the avg log-transformed peak amplitudes from each slice
    """

    mean_amps = (
        amps.groupby(["Timepoint", "Slice", "Cell type"])[
            "Mean trace peak (pA)", "Log mean trace peak"
        ]
        .mean()
        .reset_index()
    )

    mean_amps.rename(
        columns={
            "Mean trace peak (pA)": "Avg Mean trace peak (pA)",
            "Log mean trace peak": "Avg log mean trace peak",
        },
        inplace=True,
    )

    mean_amps = mean_amps.pivot(
        index=["Timepoint", "Slice"],
        columns="Cell type",
        values="Avg log mean trace peak",
    )

    mean_amps.reset_index(inplace=True)

    return mean_amps


def get_all_amp_pairs(amps):
    """
    Gets all the pairs of mean trace peak amplitudes for MCs and TCs from
    the same slice and calculates TC/MC ratio
    """
    timepoints = ["p2", "p14"]
    all_ratios = pd.DataFrame()
    for ct, timepoint in enumerate(timepoints):
        timepoint_df = amps.loc[amps["Timepoint"] == timepoint]
        slices = timepoint_df["Slice"].unique()
        timepoint_ratios = pd.DataFrame()

        for slice in slices:

            slice_list = []
            mc_names_list = []
            tc_names_list = []
            mc_amp_list = []
            tc_amp_list = []
            ratio_list = []

            timepoint_df.groupby(["Slice", "Cell type"])["Log mean trace peak"]
            slice_df = timepoint_df.loc[timepoint_df["Slice"] == slice]

            mc_list = slice_df.loc[slice_df["Cell type"] == "MC"]["Cell name"]
            tc_list = slice_df.loc[slice_df["Cell type"] == "TC"]["Cell name"]

            slice_cell_types = slice_df["Cell type"].unique().tolist()

            if "MC" in slice_cell_types and "TC" in slice_cell_types:
                pairs = list(itertools.product(mc_list, tc_list))
                for pair in pairs:
                    mc_name = pair[0]
                    tc_name = pair[1]

                    mc_amp = float(
                        timepoint_df.loc[
                            (timepoint_df["Slice"] == slice)
                            & (timepoint_df["Cell name"] == mc_name)
                        ]["Mean trace peak (pA)"]
                    )

                    tc_amp = float(
                        timepoint_df.loc[
                            (timepoint_df["Slice"] == slice)
                            & (timepoint_df["Cell name"] == tc_name)
                        ]["Mean trace peak (pA)"]
                    )

                    ratio = abs(tc_amp) / abs(mc_amp)

                    slice_list.append(slice)
                    mc_names_list.append(mc_name)
                    tc_names_list.append(tc_name)
                    mc_amp_list.append(mc_amp)
                    tc_amp_list.append(tc_amp)
                    ratio_list.append(ratio)

                pairs_amps = pd.DataFrame(
                    {
                        "Slice": slice_list,
                        "MC cell name": mc_names_list,
                        "TC cell name": tc_names_list,
                        "MC amplitude (pA)": mc_amp_list,
                        "TC amplitude (pA)": tc_amp_list,
                        "TC/MC ratio": ratio_list,
                    },
                )

                timepoint_ratios = pd.concat([timepoint_ratios, pairs_amps])
                timepoint_ratios["Timepoint"] = timepoint
        all_ratios = pd.concat([all_ratios, timepoint_ratios])

    return all_ratios


def count_ratios(ratios_df):
    """
    For each timepoint, counts the proportion of TC/MC mean trace peak amp
    ratios that are less than and greater than 1
    """
    ratios_counts = pd.DataFrame()

    for timepoint in ratios_df["Timepoint"].unique():
        total = len(
            ratios_df.loc[ratios_df["Timepoint"] == timepoint]["TC/MC ratio"]
        )
        tc_smaller = sum(
            ratios_df.loc[ratios_df["Timepoint"] == timepoint]["TC/MC ratio"]
            < 1
        )
        tc_bigger = total - tc_smaller

        timepoint_counts = pd.DataFrame(
            {
                "timepoint": timepoint,
                "total": total,
                "MC > TC": tc_smaller,
                "MC < TC": tc_bigger,
            },
            index=[0],
        )
        ratios_counts = pd.concat([ratios_counts, timepoint_counts])

    return ratios_counts

