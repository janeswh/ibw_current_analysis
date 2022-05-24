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

# from scipy.stats import sem
# from scipy import stats
import scipy
import collections

import plotly.io as pio

pio.renderers.default = "browser"

import pingouin as pg  # stats package

import p2_acq_parameters
import p14_acq_parameters
import pdb
from single_test import JaneCell
from file_settings import FileSettings
from plotting import *


class BothConditions(object):
    def __init__(self, dataset, csvfile, cell_name):
        self.dataset = dataset
        self.csvfile = csvfile
        self.cell_name = cell_name

        self.light_sweeps = None
        self.spon_sweeps = None

        self.cell_type = None
        self.stim_time = None
        self.post_stim = None

        self.amplitude_hist = self.rise_time_hist = self.tau_hist = None
        self.both_freqs_fig = None

    def get_both_conditions(self):
        """
        Gets the cell objects for both conditions and sets parameters
        """
        file_names = [
            f"{cell_name}_light100.ibw",
            f"{cell_name}_spontaneous.ibw",
        ]

        self.light_sweeps = run_single(
            self.dataset, self.csvfile, file_names[0]
        )
        self.spon_sweeps = run_single(
            self.dataset, self.csvfile, file_names[1]
        )

        self.cell_type = self.light_sweeps.cell_type
        self.stim_time = self.light_sweeps.stim_time
        self.response_window_end = self.light_sweeps.response_window_end

        self.tables_folder = (
            f"{FileSettings.TABLES_FOLDER}/{self.dataset}/"
            f"{self.cell_type}/{self.cell_name}/"
        )

        self.figs_folder = (
            f"{FileSettings.FIGURES_FOLDER}/{self.dataset}/"
            f"{self.cell_type}/{self.cell_name}/"
        )

        if not os.path.exists(self.tables_folder):
            os.makedirs(self.tables_folder)

        if not os.path.exists(self.figs_folder):
            os.makedirs(self.figs_folder)

    def save_freqs(self):
        """
        Saves the raw frequency and average frequencies from each condition
        """

        raw_freqs_df = pd.DataFrame(
            {
                "Light Raw Frequency (Hz)": self.light_sweeps.freq.iloc[:, 0],
                "Spontaneous Raw Frequency (Hz)": self.spon_sweeps.freq.iloc[
                    :, 0
                ],
            },
            index=self.light_sweeps.freq.index,
        )

        avg_freqs_df = pd.DataFrame(
            {
                "Light Avg Frequency (Hz)": self.light_sweeps.avg_frequency_df.iloc[
                    :, 0
                ],
                "Spontaneous Avg Frequency (Hz)": self.spon_sweeps.avg_frequency_df.iloc[
                    :, 0
                ],
            },
            index=self.light_sweeps.avg_frequency_df.index,
        )

        file_names = [
            f"{self.cell_name}_raw_frequency.csv",
            f"{self.cell_name}_avg_frequency.csv",
        ]
        dfs = [raw_freqs_df, avg_freqs_df]

        for count, file_name in enumerate(file_names):
            path = os.path.join(self.tables_folder, file_name)
            dfs[count].to_csv(path, float_format="%8.4f", index=True)

    def save_avg_freq_stats(self):
        """
        Saves the kinetic stats of the average frequency for both conditions
        """

        combined_df = pd.concat(
            [
                self.light_sweeps.avg_frequency_stats,
                self.spon_sweeps.avg_frequency_stats,
            ]
        )
        combined_df.index = ["Light", "Spontaneous"]

        file_name = f"{self.cell_name}_avg_freq_stats.csv"
        path = os.path.join(self.tables_folder, file_name)

        combined_df.to_csv(path, float_format="%8.4f", index=True)

    def save_event_stats(self):
        """
        Saves individual event stats for both conditions, one file each for
        within response window (500-2000 ms) and outside
        """

        # drops rise start and end columns
        cols_to_drop = ["Rise start (ms)", "Rise end (ms)", "Root time (ms)"]
        light_stats = self.light_sweeps.event_stats.drop(
            labels=cols_to_drop, axis=1
        )
        spon_stats = self.spon_sweeps.event_stats.drop(
            labels=cols_to_drop, axis=1
        )

        light_response_win_stats = light_stats[
            light_stats["New pos"].between(
                self.stim_time, self.response_window_end, inclusive="both"
            )
        ]
        light_outside_win_stats = light_stats[
            ~light_stats["New pos"].between(
                self.stim_time, self.response_window_end, inclusive="both"
            )
        ]

        spon_response_win_stats = spon_stats[
            spon_stats["New pos"].between(
                self.stim_time, self.response_window_end, inclusive="both"
            )
        ]
        spon_outside_win_stats = spon_stats[
            ~spon_stats["New pos"].between(
                self.stim_time, self.response_window_end, inclusive="both"
            )
        ]

        file_names = [
            f"{self.cell_name}_window_light_event_stats.csv",
            f"{self.cell_name}_window_spontaneous_event_stats.csv",
            f"{self.cell_name}_outside_light_event_stats.csv",
            f"{self.cell_name}_outside_spontaneous_event_stats.csv",
        ]
        dfs = [
            light_response_win_stats,
            spon_response_win_stats,
            light_outside_win_stats,
            spon_outside_win_stats,
        ]

        for count, file_name in enumerate(file_names):
            path = os.path.join(self.tables_folder, file_name)
            dfs[count].to_csv(path, float_format="%8.4f", index=True)

        # saves medians from both conditions in one csv, for within window
        window_median_event_stats = pd.DataFrame(
            {
                "Light": light_response_win_stats.median(),
                "Spontaneous": spon_response_win_stats.median(),
            },
            index=light_response_win_stats.columns,
        )

        window_median_event_stats.drop(
            labels=["Sweep", "New pos"], axis=0, inplace=True
        )

        window_median_stats_file_name = (
            f"{self.cell_name}_median_window_" f"event_stats.csv"
        )
        window_median_stats_path = os.path.join(
            self.tables_folder, window_median_stats_file_name
        )
        window_median_event_stats.to_csv(
            window_median_stats_path, float_format="%8.4f", index=True
        )

        # saves medians from both conditions in one csv, for outside window
        outside_median_event_stats = pd.DataFrame(
            {
                "Light": light_outside_win_stats.median(),
                "Spontaneous": spon_outside_win_stats.median(),
            },
            index=light_outside_win_stats.columns,
        )

        outside_median_event_stats.drop(
            labels=["Sweep", "New pos"], axis=0, inplace=True
        )

        outside_median_stats_file_name = (
            f"{self.cell_name}_median_outside_event_stats.csv"
        )
        outside_median_stats_path = os.path.join(
            self.tables_folder, outside_median_stats_file_name
        )
        outside_median_event_stats.to_csv(
            outside_median_stats_path, float_format="%8.4f", index=True
        )

    def save_mean_trace_stats(self):
        """
        Saves mean trace peak amplitude and peak time
        """

        combined_df = pd.concat(
            [
                self.light_sweeps.mean_trace_stats,
                self.spon_sweeps.mean_trace_stats,
            ]
        )
        combined_df.index = ["Light", "Spontaneous"]

        file_name = f"{self.cell_name}_mean_trace_stats.csv"
        path = os.path.join(self.tables_folder, file_name)

        combined_df.to_csv(path, float_format="%8.4f", index=True)

    def save_csvs(self):
        """
        Saves the frequency and stats values from running individual conditions
        """

        self.save_freqs()
        self.save_avg_freq_stats()
        self.save_event_stats()
        self.save_mean_trace_stats()

    def plot_event_stats_histograms(self):
        """
        Reads in saved csvs and makes histograms for each event kinetic - 
        amplitude, rise time, tau
        """

        light_win_stats_file = os.path.join(
            self.tables_folder,
            f"{self.cell_name}_window_light_event_stats.csv",
        )
        light_win_stats_df = pd.read_csv(light_win_stats_file, index_col=0)

        light_outside_stats_file = os.path.join(
            self.tables_folder,
            f"{self.cell_name}_outside_light_event_stats.csv",
        )
        light_outside_stats_df = pd.read_csv(
            light_outside_stats_file, index_col=0
        )

        spon_win_stats_file = os.path.join(
            self.tables_folder,
            f"{self.cell_name}_window_spontaneous_event_stats.csv",
        )
        spon_win_stats_df = pd.read_csv(spon_win_stats_file, index_col=0)

        spon_outside_stats_file = os.path.join(
            self.tables_folder,
            f"{self.cell_name}_outside_spontaneous_event_stats.csv",
        )
        spon_outside_stats_df = pd.read_csv(
            spon_outside_stats_file, index_col=0
        )

        dfs_lists = [
            [light_win_stats_df, light_outside_stats_df],
            [spon_win_stats_df, spon_outside_stats_df],
        ]
        conditions = ["Light", "Spontaneous"]
        colors = ["#B958F2", "#8A8C89"]
        median_colors = ["#D49BF5", "#B9B9B9"]

        # makes the three subplots, one for each measurement
        amplitude_hist = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=(
                "Response Window Events",
                "Outside Response Window Events",
            ),
        )

        rise_time_hist = make_subplots(
            rows=1, cols=2, shared_xaxes=True, shared_yaxes=True
        )

        tau_hist = make_subplots(
            rows=1, cols=2, shared_xaxes=True, shared_yaxes=True
        )

        for count, df_list in enumerate(dfs_lists):

            for win_count, condition_stats in enumerate(df_list):

                # adds traces to amplitude histogram fig

                amp_trace = make_event_hist_trace(
                    colors[count],
                    conditions[count],
                    condition_stats["New amplitude (pA)"],
                )
                amplitude_hist.add_trace(amp_trace, row=1, col=win_count + 1)
                add_median_vline(
                    amplitude_hist,
                    median_colors[count],
                    condition_stats["New amplitude (pA)"],
                    win_count,
                    count,
                    "pA",
                )

                amplitude_hist.update_xaxes(
                    title="Amplitude (pA)",
                    autorange="reversed",
                    row=1,
                    col=win_count + 1,
                    matches="x",
                )

                # adds traces to rise time histogram fig
                rise_time_trace = make_event_hist_trace(
                    colors[count],
                    conditions[count],
                    condition_stats["Rise time (ms)"],
                )
                rise_time_hist.add_trace(
                    rise_time_trace, row=1, col=win_count + 1
                )
                add_median_vline(
                    rise_time_hist,
                    median_colors[count],
                    condition_stats["Rise time (ms)"],
                    win_count,
                    count,
                    "ms",
                )

                rise_time_hist.update_xaxes(
                    title="Rise time (ms)",
                    row=1,
                    col=win_count + 1,
                    matches="x",
                )

                # adds traces to tau histogram fig
                tau_trace = make_event_hist_trace(
                    colors[count],
                    conditions[count],
                    condition_stats["Tau (ms)"],
                )
                tau_hist.add_trace(tau_trace, row=1, col=win_count + 1)
                add_median_vline(
                    tau_hist,
                    median_colors[count],
                    condition_stats["Tau (ms)"],
                    win_count,
                    count,
                    "ms",
                )

                tau_hist.update_xaxes(
                    title="Tau (ms)", row=1, col=win_count + 1, matches="x"
                )

        hist_list = [amplitude_hist, rise_time_hist, tau_hist]
        for hist in hist_list:
            # below is code from stack overflow to hide duplicate legends
            names = set()
            hist.for_each_trace(
                lambda trace: trace.update(showlegend=False)
                if (trace.name in names)
                else names.add(trace.name)
            )

        self.amplitude_hist = amplitude_hist
        self.rise_time_hist = rise_time_hist
        self.tau_hist = tau_hist

    def plot_both_freqs(self):

        avg_freq_csv_file = os.path.join(
            self.tables_folder, f"{self.cell_name}_avg_frequency.csv"
        )
        avg_freqs_df = pd.read_csv(avg_freq_csv_file, index_col=0)

        raw_freq_csv_file = os.path.join(
            self.tables_folder, f"{self.cell_name}_raw_frequency.csv"
        )
        raw_freqs_df = pd.read_csv(raw_freq_csv_file, index_col=0)

        both_freqs_fig = go.Figure()

        # plots histogram
        both_freqs_fig.add_trace(
            go.Bar(
                x=raw_freqs_df.index,
                y=raw_freqs_df["Light Raw Frequency (Hz)"],
                marker=dict(color="#D9ABF4"),
                name="light raw frequency",
            )
        )

        both_freqs_fig.add_trace(
            go.Bar(
                x=raw_freqs_df.index,
                y=raw_freqs_df["Spontaneous Raw Frequency (Hz)"],
                marker=dict(color="#B7B9B5"),
                name="Spontaneous raw frequency",
            )
        )

        # this removes the white outline of the bar graph to emulate histogram
        both_freqs_fig.update_traces(marker=dict(line=dict(width=0)),)

        # removes gaps between bars, puts traces ontop of each other
        both_freqs_fig.update_layout(barmode="overlay", bargap=0)

        # plots avg frequency smoothed trace
        both_freqs_fig.add_trace(
            go.Scatter(
                x=avg_freqs_df.index,
                y=avg_freqs_df["Light Avg Frequency (Hz)"],
                marker=dict(color="#B958F2"),
                name="Light avg frequency",
            )
        )

        both_freqs_fig.add_trace(
            go.Scatter(
                x=avg_freqs_df.index,
                y=avg_freqs_df["Spontaneous Avg Frequency (Hz)"],
                marker=dict(color="#8A8C89"),
                name="Spontaneous avg frequency",
            )
        )

        # adds blue overlay to show light stim duration
        both_freqs_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        # update axes titles
        both_freqs_fig.update_yaxes(title_text="Frequency (Hz)")
        both_freqs_fig.update_xaxes(title_text="Time (ms)")

        # add main title, x-axis titles
        both_freqs_fig.update_layout(
            title_text="{}, {} Frequency Comparisons".format(
                self.cell_name, self.cell_type
            ),
            title_x=0.5,
        )

        # both_freqs_fig.show()

        self.both_freqs_fig = both_freqs_fig

    def save_freq_event_stats_plots(self):
        """
        Saves frequency plot and event stats plots as one html file
        """

        html_filename = f"{self.cell_name}_stats_plots.html"
        path = os.path.join(self.figs_folder, html_filename)

        self.both_freqs_fig.write_html(
            path, full_html=False, include_plotlyjs="cdn"
        )

        for hist in [self.amplitude_hist, self.rise_time_hist, self.tau_hist]:
            with open(path, "a") as f:
                f.write(hist.to_html(full_html=False, include_plotlyjs=False))


# def get_both_conditions(dataset, csvfile, cell_name):
#     """
#     Should do this to avoid repeatedly running analysis
#     1. get cell object for each condition
#     2. save all the data to csv
#     3. import data and plot
#     4. save plots
#     """

#     file_names = [
#         f"{cell_name}_light100.ibw",
#         f"{cell_name}_spontaneous.ibw",
#     ]

#     light_sweeps = run_single(dataset, csvfile, file_names[0])
#     spon_sweeps = run_single(dataset, csvfile, file_names[1])

#     cell_type = light_sweeps.cell_type
#     stim_time = light_sweeps.stim_time

#     # should save avg frequency traces to csv so we don't have to run
#     # the bayesian stuff every time

#     # also save avg frequency stats
#     # save event stats
#     save_freqs(
#         dataset,
#         cell_type,
#         cell_name,
#         light_sweeps.freq,
#         light_sweeps.avg_frequency_df,
#         spon_sweeps.freq,
#         spon_sweeps.avg_frequency_df,
#     )

#     save_avg_freq_stats(
#         dataset,
#         cell_type,
#         cell_name,
#         light_sweeps.avg_frequency_stats,
#         spon_sweeps.avg_frequency_stats,
#     )

#     save_event_stats(
#         dataset,
#         cell_type,
#         cell_name,
#         light_sweeps.event_stats,
#         spon_sweeps.event_stats,
#     )

#     save_mean_trace_stats(
#         dataset,
#         cell_type,
#         cell_name,
#         light_sweeps.mean_trace_stats,
#         spon_sweeps.mean_trace_stats,
#     )

#     amplitude_hist, rise_time_hist, tau_hist = plot_event_stats(
#         dataset, cell_name, cell_type
#     )

#     hist_list = [amplitude_hist, rise_time_hist, tau_hist]

#     freqs_fig = plot_both_freqs(dataset, cell_name, cell_type, stim_time)

#     output_html_plots(
#         hist_list, freqs_fig, dataset, cell_type, cell_name,
#     )

#     response = check_response(dataset, cell_type, cell_name)
#     if response is True:
#         print("cell has response")


# def check_response(dataset, cell_type, cell_name):
#     # determine whether cell has response with KS test
#     p_value = run_KS_test(dataset, cell_type, cell_name)

#     if p_value <= 0.05:
#         response = True
#     else:
#         response = False

#     return response


# def save_mean_trace_stats(
#     dataset, cell_type, cell_name, light_stats, spon_stats
# ):
#     combined_df = pd.concat([light_stats, spon_stats])
#     combined_df.index = ["Light", "Spontaneous"]

#     base_path = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
#     )
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#     file_name = f"{cell_name}_mean_trace_stats.csv"
#     path = os.path.join(base_path, file_name)

#     combined_df.to_csv(path, float_format="%8.4f", index=True)


# def save_avg_freq_stats(
#     dataset, cell_type, cell_name, light_stats, spon_stats
# ):
#     combined_df = pd.concat([light_stats, spon_stats])
#     combined_df.index = ["Light", "Spontaneous"]

#     base_path = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
#     )
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#     file_name = f"{cell_name}_avg_freq_stats.csv"
#     path = os.path.join(base_path, file_name)

#     combined_df.to_csv(path, float_format="%8.4f", index=True)


# def save_event_stats(dataset, cell_type, cell_name, light_stats, spon_stats):

#     # saves individual event stats

#     # drops rise start and end columns
#     cols_to_drop = ["Rise start (ms)", "Rise end (ms)", "Root time (ms)"]
#     light_stats.drop(labels=cols_to_drop, axis=1, inplace=True)
#     spon_stats.drop(labels=cols_to_drop, axis=1, inplace=True)

#     base_path = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
#     )
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)

#     file_names = [
#         f"{cell_name}_light_event_stats.csv",
#         f"{cell_name}_spontaneous_event_stats.csv",
#     ]
#     dfs = [light_stats, spon_stats]

#     for count, file_name in enumerate(file_names):
#         path = os.path.join(base_path, file_name)
#         dfs[count].to_csv(path, float_format="%8.4f", index=True)

#     # saves medians from both conditions in one csv
#     median_event_stats = pd.DataFrame(
#         {"Light": light_stats.mean(), "Spontaneous": spon_stats.median()},
#         index=light_stats.columns,
#     )

#     median_event_stats.drop(labels=["Sweep", "New pos"], axis=0, inplace=True)

#     median_stats_file_name = f"{cell_name}_median_event_stats.csv"
#     median_stats_path = os.path.join(base_path, median_stats_file_name)
#     median_event_stats.to_csv(
#         median_stats_path, float_format="%8.4f", index=True
#     )


# def save_freqs(
#     dataset,
#     cell_type,
#     cell_name,
#     light_raw_freq,
#     light_avg_freq,
#     spon_raw_freq,
#     spon_avg_freq,
# ):

#     raw_freqs_df = pd.DataFrame(
#         {
#             "Light Raw Frequency (Hz)": light_raw_freq.iloc[:, 0],
#             "Spontaneous Raw Frequency (Hz)": spon_raw_freq.iloc[:, 0],
#         },
#         index=light_raw_freq.index,
#     )

#     avg_freqs_df = pd.DataFrame(
#         {
#             "Light Avg Frequency (Hz)": light_avg_freq.iloc[:, 0],
#             "Spontaneous Avg Frequency (Hz)": spon_avg_freq.iloc[:, 0],
#         },
#         index=light_avg_freq.index,
#     )

#     base_path = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
#     )
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)

#     file_names = [
#         f"{cell_name}_raw_frequency.csv",
#         f"{cell_name}_avg_frequency.csv",
#     ]
#     dfs = [raw_freqs_df, avg_freqs_df]

#     for count, file_name in enumerate(file_names):
#         path = os.path.join(base_path, file_name)
#         dfs[count].to_csv(path, float_format="%8.4f", index=True)


def run_KS_test(dataset, cell_type, cell_name):
    """
    Runs two-sample Kolmogorov-Smirnov test and returns p-value
    """
    freqs_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_avg_frequency.csv"
    )
    avg_freqs = pd.read_csv(freqs_file, index_col=0)

    if dataset == "p2":
        stim_time = p2_acq_parameters.STIM_TIME
    if dataset == "p14":
        stim_time = p14_acq_parameters.STIM_TIME

    light_freq = avg_freqs["Light Avg Frequency (Hz)"][stim_time:]
    spon_freq = avg_freqs["Spontaneous Avg Frequency (Hz)"][stim_time:]

    # subtract the baseline from the averages???
    freq_stats_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_avg_freq_stats.csv"
    )
    freq_stats = pd.read_csv(freq_stats_file, index_col=0)

    light_baseline = freq_stats.loc["Light"]["Baseline Frequency (Hz)"]
    spon_baseline = freq_stats.loc["Spontaneous"]["Baseline Frequency (Hz)"]

    conditions = ["Light", "Spontaneous"]
    condition_traces = [light_freq, spon_freq]
    baselines = [light_baseline, spon_baseline]

    win_indices = [
        slice(light_freq.index[0], light_freq.index[-1]),
        slice(stim_time, 1000),
    ]

    nested_dict = lambda: defaultdict(nested_dict)
    nest_pval_dict = nested_dict()

    subtracts = ["no sub", "sub avg", "sub baseline"]
    for window in win_indices:
        if window == slice(light_freq.index[0], light_freq.index[-1]):
            window_label = "whole window"
        else:
            window_label = "500-1000 ms"
        for subtract_type in subtracts:
            ttest_stats, ttest_pval, ks_stats, ks_pval = run_stats(
                condition_traces, window, baselines, subtract=subtract_type
            )

            nest_pval_dict[window_label]["ttest stats"][
                subtract_type
            ] = ttest_stats

            nest_pval_dict[window_label]["ttest pval"][
                subtract_type
            ] = ttest_pval
            nest_pval_dict[window_label]["ks stats"][subtract_type] = ks_stats
            nest_pval_dict[window_label]["ks pval"][subtract_type] = ks_pval

    # plot traces to see what's going on
    colors = ["#B958F2", "#8A8C89"]

    plot_response_win_comparison(
        cell_type,
        cell_name,
        stim_time,
        colors,
        conditions,
        condition_traces,
        win_indices,
        baselines,
        nest_pval_dict,
    )


def run_stats(condition_traces, window, baselines, subtract):
    """
    Runs t-test and KS test on two arrays, returning p-values. Also performs
    average or baseline subtractions prior to running test, as needed
    """
    x = condition_traces[0][window]
    y = condition_traces[1][window]

    if subtract == "sub avg":
        x = x - x.mean()
        y = y - y.mean()

    elif subtract == "sub baseline":
        x = x - baselines[0]
        y = y - baselines[1]

    ttest_stats = pg.ttest(x, y)
    ttest_pval = np.round(ttest_stats["p-val"][0], 5)
    # ttest_pval = f"{ttest_pval:.2e}"

    ks_stats, ks_pval = scipy.stats.ks_2samp(x, y)
    # ks_pval = f"{ks_pval:.2e}"
    ks_pval = np.round(ks_pval, 5)

    return ttest_stats, ttest_pval, ks_stats, ks_pval


# def plot_event_stats(dataset, cell_name, cell_type):

#     light_stats_file = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
#         f"{cell_name}_light_event_stats.csv"
#     )
#     light_stats_df = pd.read_csv(light_stats_file, index_col=0)

#     spon_stats_file = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
#         f"{cell_name}_spontaneous_event_stats.csv"
#     )
#     spon_stats_df = pd.read_csv(spon_stats_file, index_col=0)

#     dfs = [light_stats_df, spon_stats_df]
#     conditions = ["Light", "Spontaneous"]
#     colors = ["#B958F2", "#8A8C89"]
#     median_colors = ["#D49BF5", "#B9B9B9"]

#     if dataset == "p2":
#         stim_time = p2_acq_parameters.STIM_TIME
#         post_time = (
#             p2_acq_parameters.RESPONSE_WINDOW_END
#         )  # time to stop looking

#     if dataset == "p14":
#         stim_time = p14_acq_parameters.STIM_TIME
#         post_time = (
#             p14_acq_parameters.RESPONSE_WINDOW_END
#         )  # time to stop looking

#     # makes the three subplots, one for each measurement
#     amplitude_hist = make_subplots(
#         rows=1,
#         cols=2,
#         shared_xaxes=True,
#         shared_yaxes=True,
#         subplot_titles=(
#             "Response Window Events",
#             "Outside Response Window Events",
#         ),
#     )

#     rise_time_hist = make_subplots(
#         rows=1, cols=2, shared_xaxes=True, shared_yaxes=True
#     )

#     tau_hist = make_subplots(
#         rows=1, cols=2, shared_xaxes=True, shared_yaxes=True
#     )

#     for count, df in enumerate(dfs):

#         response_win_stats = df[
#             df["New pos"].between(stim_time, post_time, inclusive="both")
#         ]
#         outside_win_stats = df[
#             ~df["New pos"].between(stim_time, post_time, inclusive="both")
#         ]

#         win_df_list = [response_win_stats, outside_win_stats]

#         for win_count, condition_stats in enumerate(win_df_list):

#             # adds traces to amplitude histogram fig

#             amp_trace = make_event_hist_trace(
#                 colors[count],
#                 conditions[count],
#                 condition_stats["New amplitude (pA)"],
#             )
#             amplitude_hist.add_trace(amp_trace, row=1, col=win_count + 1)
#             add_median_vline(
#                 amplitude_hist,
#                 median_colors[count],
#                 condition_stats["New amplitude (pA)"],
#                 win_count,
#                 count,
#                 "pA",
#             )

#             amplitude_hist.update_xaxes(
#                 title="Amplitude (pA)",
#                 autorange="reversed",
#                 row=1,
#                 col=win_count + 1,
#             )

#             # adds traces to rise time histogram fig
#             rise_time_trace = make_event_hist_trace(
#                 colors[count],
#                 conditions[count],
#                 condition_stats["Rise time (ms)"],
#             )
#             rise_time_hist.add_trace(rise_time_trace, row=1, col=win_count + 1)
#             add_median_vline(
#                 rise_time_hist,
#                 median_colors[count],
#                 condition_stats["Rise time (ms)"],
#                 win_count,
#                 count,
#                 "ms",
#             )

#             rise_time_hist.update_xaxes(
#                 title="Rise time (ms)", row=1, col=win_count + 1,
#             )

#             # adds traces to tau histogram fig
#             tau_trace = make_event_hist_trace(
#                 colors[count], conditions[count], condition_stats["Tau (ms)"],
#             )
#             tau_hist.add_trace(tau_trace, row=1, col=win_count + 1)
#             add_median_vline(
#                 tau_hist,
#                 median_colors[count],
#                 condition_stats["Tau (ms)"],
#                 win_count,
#                 count,
#                 "ms",
#             )

#             tau_hist.update_xaxes(
#                 title="Tau (ms)", row=1, col=win_count + 1,
#             )

#     hist_list = [amplitude_hist, rise_time_hist, tau_hist]
#     for hist in hist_list:
#         # below is code from stack overflow to hide duplicate legends
#         names = set()
#         hist.for_each_trace(
#             lambda trace: trace.update(showlegend=False)
#             if (trace.name in names)
#             else names.add(trace.name)
#         )

#     return amplitude_hist, rise_time_hist, tau_hist


# def plot_both_freqs(dataset, cell_name, cell_type, stim_time):

#     avg_freq_csv_file = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
#         f"{cell_name}_avg_frequency.csv"
#     )
#     avg_freqs_df = pd.read_csv(avg_freq_csv_file, index_col=0)

#     raw_freq_csv_file = (
#         f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
#         f"{cell_name}_raw_frequency.csv"
#     )
#     raw_freqs_df = pd.read_csv(raw_freq_csv_file, index_col=0)

#     both_freqs_fig = go.Figure()

#     # plots histogram
#     both_freqs_fig.add_trace(
#         go.Bar(
#             x=raw_freqs_df.index,
#             y=raw_freqs_df["Light Raw Frequency (Hz)"],
#             marker=dict(color="#D9ABF4"),
#             name="light raw frequency",
#         )
#     )

#     both_freqs_fig.add_trace(
#         go.Bar(
#             x=raw_freqs_df.index,
#             y=raw_freqs_df["Spontaneous Raw Frequency (Hz)"],
#             marker=dict(color="#B7B9B5"),
#             name="Spontaneous raw frequency",
#         )
#     )

#     # this removes the white outline of the bar graph to emulate histogram
#     both_freqs_fig.update_traces(marker=dict(line=dict(width=0)),)

#     # removes gaps between bars, puts traces ontop of each other
#     both_freqs_fig.update_layout(barmode="overlay", bargap=0)

#     # plots avg frequency smoothed trace
#     both_freqs_fig.add_trace(
#         go.Scatter(
#             x=avg_freqs_df.index,
#             y=avg_freqs_df["Light Avg Frequency (Hz)"],
#             marker=dict(color="#B958F2"),
#             name="Light avg frequency",
#         )
#     )

#     both_freqs_fig.add_trace(
#         go.Scatter(
#             x=avg_freqs_df.index,
#             y=avg_freqs_df["Spontaneous Avg Frequency (Hz)"],
#             marker=dict(color="#8A8C89"),
#             name="Spontaneous avg frequency",
#         )
#     )

#     # adds blue overlay to show light stim duration
#     both_freqs_fig.add_vrect(
#         x0=stim_time,
#         x1=stim_time + 100,
#         fillcolor="#33F7FF",
#         opacity=0.5,
#         layer="below",
#         line_width=0,
#     )

#     # update axes titles
#     both_freqs_fig.update_yaxes(title_text="Frequency (Hz)")
#     both_freqs_fig.update_xaxes(title_text="Time (ms)")

#     # add main title, x-axis titles
#     both_freqs_fig.update_layout(
#         title_text="{}, {} Frequency Comparisons".format(cell_name, cell_type),
#         title_x=0.5,
#     )

#     # both_freqs_fig.show()

#     return both_freqs_fig


# def output_html_plots(
#     hists, freqs_fig, dataset, cell_type, cell_name,
# ):
#     """
#     Saves event stats and avg freq stats figs as one html plot
#     """
#     base_path = (
#         f"{FileSettings.FIGURES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
#     )
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)

#     html_filename = f"{cell_name}_stats_plots.html"
#     path = os.path.join(base_path, html_filename)

#     freqs_fig.write_html(path, full_html=False, include_plotlyjs="cdn")

#     for hist in hists:
#         with open(path, "a") as f:
#             f.write(hist.to_html(full_html=False, include_plotlyjs=False))


def run_single(dataset, csvfile, file_name):
    file = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/data",
        dataset,
        file_name,
    )

    # gets sweep info for all cells
    sweep_info = pd.read_csv(csvfile, index_col=0)

    # 0 initializes JaneCell class
    condition_sweeps = JaneCell(dataset, sweep_info, file, file_name)

    # 1 checks whether cell has a response before proceeding
    response = condition_sweeps.check_response()

    # 4 runs stats on sweeps and creates a dict for each stim condition
    condition_sweeps.get_mod_events()
    condition_sweeps.calculate_event_stats()
    # condition_sweeps.plot_mod_events  # sanity check only
    condition_sweeps.calculate_mean_trace_stats()
    condition_sweeps.plot_annotated_events()
    condition_sweeps.save_annotated_events_plot()

    # cell.plot_events()    # sanity check only
    condition_sweeps.analyze_avg_frequency()
    condition_sweeps.save_annotated_freq()

    condition_sweeps.plot_mean_trace()
    # cell.save_mean_trace_plot()   # don't save, each plot is 130 mb

    return condition_sweeps


def run_both_conditions(dataset, csvfile, cell_name):
    """
    Analyzes the light and spontaneous ibws for each cell
    """
    # 0 initializes BothConditions class
    cell = BothConditions(dataset, csvfile, cell_name)

    # 1 runs analysis for each set of condition sweeps
    cell.get_both_conditions()

    # 2 saves frequency and event stats data to csv
    cell.save_csvs()

    # 3 plots and saves comparisons between light vs. spontaneous
    cell.plot_event_stats_histograms()
    cell.plot_both_freqs()
    cell.save_freq_event_stats_plots()

    pdb.set_trace()


if __name__ == "__main__":
    # dataset = "p2"
    dataset = "p14"
    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables",
        dataset,
        csvfile_name,
    )
    # cell_name = "JH200311_c1"
    cell_name = "JH190905_c7"

    run_both_conditions(dataset, csvfile, cell_name)

    # file_name = "JH200303_c7_light100.ibw"

    # run_single(dataset, csvfile, file_name)

    print("Analysis for {} done".format(cell_name))
