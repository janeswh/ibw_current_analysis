import pandas as pd
import matplotlib.pyplot as plt
import pynwb
import os
import numpy as np
from neo.io import IgorIO
from pynwb import NWBHDF5IO
import elephant
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import sem
from scipy.stats import (
    anderson_ksamp,
    ks_2samp,
    pearsonr,
    spearmanr,
    ttest_ind,
)

import plotly.io as pio

pio.renderers.default = "browser"

from p2_acq_parameters import *
from p14_acq_parameters import *
import pdb

from plotting import *
from aggregate_stats import *


def do_stats(csvfile_name, measures_list, test_type):
    csvfile = os.path.join(
        FileSettings.TABLES_FOLDER,
        "datasets_summaries_data",
        csvfile_name,
    )

    df = pd.read_csv(csvfile, index_col=0, header=0)

    timepoints = df["Dataset"].unique()

    stats = pd.DataFrame()
    print(f"Analyzing {test_type} stats for {csvfile_name}")
    for timepoint in timepoints:
        for measure in measures_list:
            mc_df = df.loc[
                (df["Cell Type"] == "MC") & (df["Dataset"] == timepoint),
                measure,
            ]
            tc_df = df.loc[
                (df["Cell Type"] == "TC") & (df["Dataset"] == timepoint),
                measure,
            ]

            if test_type == "Anderson-Darling":
                anderson_statistic, crit_vals, p_val = anderson_ksamp(
                    [mc_df, tc_df]
                )
            elif test_type == "KS":
                ks_statistics, p_val = ks_2samp(mc_df, tc_df)

            elif test_type == "2-samp_ttest":
                ttest_statistic, p_val = ttest_ind(mc_df, tc_df)

            print(
                f"{test_type} p-val for {timepoint} cell type comparison "
                f"of {measure} = {np.round(p_val,3)}"
            )

            stats_list = pd.DataFrame(
                [csvfile_name, test_type, timepoint, measure, p_val]
            ).T
            stats_list.columns = [
                "File",
                "Test",
                "Timepoint",
                "Measure",
                "p-value",
            ]
            stats = pd.concat([stats, stats_list])

    return stats


def do_paired_ratio_stats(csvfile_name, test_type):
    csvfile = os.path.join(
        FileSettings.TABLES_FOLDER,
        "paper_figs_data",
        csvfile_name,
    )

    df = pd.read_csv(csvfile, index_col=0, header=0)
    stats = pd.DataFrame()

    p2_ratios = df.loc[df["Timepoint"] == "P2"]["TC/MC ratio"]
    p14_ratios = df.loc[df["Timepoint"] == "P14"]["TC/MC ratio"]

    if test_type == "Anderson-Darling":
        anderson_statistic, crit_vals, p_val = anderson_ksamp(
            [p2_ratios, p14_ratios]
        )

    stats = pd.DataFrame([csvfile_name, test_type, p_val]).T
    stats.columns = [
        "File",
        "Test",
        "p-value",
    ]

    return stats


def calc_spearman(df, data_type, x_label, y_label):
    """
    Calculates the Pearson's correlation coefficient for both timepoints
    and both cell types' frequency/event amplitude and rise time
    """
    timepoints = df["Dataset"].unique()
    cell_types = df["Cell Type"].unique()

    df = df.copy()
    # reverses amplitude for correlation
    if data_type == "event":
        df["Adjusted amplitude (pA)"] = df["Adjusted amplitude (pA)"] * -1

    stats = pd.DataFrame()
    print(f"Analyzing Pearson's for {data_type}")
    for timepoint in timepoints:
        for cell_type in cell_types:
            x_array = df.loc[
                (df["Cell Type"] == cell_type) & (df["Dataset"] == timepoint),
                x_label,
            ]

            y_array = df.loc[
                (df["Cell Type"] == cell_type) & (df["Dataset"] == timepoint),
                y_label,
            ]

            r, p_val = spearmanr(x_array, y_array, nan_policy="omit")

            print(
                f"Spearman correlation = {np.round(r,3)},  p-val =  "
                f"{np.round(p_val,5)} for {timepoint} {cell_type}"
            )

            stats_list = pd.DataFrame(
                [data_type, timepoint, cell_type, x_label, y_label, r, p_val]
            ).T
            stats_list.columns = [
                "Data type",
                "Timepoint",
                "Cell type",
                "X value",
                "Y value",
                "Spearman r",
                "p-value",
            ]
            stats = pd.concat([stats, stats_list])

    return stats


def main():
    # stats for TC/MC mean trace peak ratios
    amp_ratio_stats = do_paired_ratio_stats(
        "paired_amp_ratios.csv", "Anderson-Darling"
    )

    # stats for response mean trace data
    mean_trace_stats = do_stats(
        "mean_trace_data.csv",
        [
            "Mean Trace Peak (pA)",
            "Log Mean Trace Peak",
            "Mean Trace Time to Peak (ms)",
        ],
        "Anderson-Darling",
    )

    # stats for all mean trace
    all_mean_trace_stats = do_stats(
        "all_mean_trace_data.csv",
        [
            "Mean Trace Peak (pA)",
            "Log Mean Trace Peak",
            "Mean Trace Time to Peak (ms)",
        ],
        "Anderson-Darling",
    )

    # stats for all mean trace, t-test
    all_mean_trace_stats = do_stats(
        "mean_trace_data.csv",
        [
            "Mean Trace Peak (pA)",
            "Log Mean Trace Peak",
            "Mean Trace Time to Peak (ms)",
        ],
        "2-samp_ttest",
    )

    # stats for freq stats
    freq_stats = do_stats(
        "freq_stats_data.csv",
        [
            "Baseline-sub Peak Freq (Hz)",
            "Time to Peak Frequency (ms)",
            "Baseline Frequency (Hz)",
            "Rise Time (ms)",
        ],
        "Anderson-Darling",
    )

    # stats for cell type comparison events
    cell_type_stats = do_stats(
        "cell_comparison_medians_data.csv",
        ["Adjusted amplitude (pA)", "Rise time (ms)", "Tau (ms)"],
        "Anderson-Darling",
    )

    all_stats = pd.concat(
        [
            amp_ratio_stats,
            mean_trace_stats,
            all_mean_trace_stats,
            freq_stats,
            cell_type_stats,
        ]
    )
    csv_filename = "anderson_darling_stats.csv"
    path = os.path.join(
        FileSettings.TABLES_FOLDER, "datasets_summaries_data", csv_filename
    )
    all_stats.to_csv(path, float_format="%8.4f")


if __name__ == "__main__":
    main()
