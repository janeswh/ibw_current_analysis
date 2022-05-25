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

        self.response_pvals = None
        self.stats_fig = None
        self.light_freq_response = None
        self.spon_freq_response = None
        self.mean_trace_response = None
        self.overall_response = None

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
            title_text=(
                f"{self.dataset} {self.cell_type} {self.cell_name} "
                f"Frequency Comparisons"
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

    def compare_avg_freqs(self):
        """
        Compares the average frequencies between light and spontaneous to 
        determine whether the cell has light-evoked response
        """
        freqs_file = os.path.join(
            self.tables_folder, f"{cell_name}_avg_frequency.csv"
        )
        avg_freqs = pd.read_csv(freqs_file, index_col=0)

        # window of comparison is different for p14 and p2 because of tp end
        # time differences  (in ms)
        if self.dataset == "p14":
            time_after_light = 90
        elif self.dataset == "p2":
            time_after_light = 100
        light_freq = avg_freqs["Light Avg Frequency (Hz)"][
            self.stim_time : self.stim_time + time_after_light
        ]
        spon_freq = avg_freqs["Spontaneous Avg Frequency (Hz)"][
            self.stim_time : self.stim_time + time_after_light
        ]

        # takes 150 ms window/15 points of pre-stim as baseline
        response_start = light_freq.index[0]

        # baseline windown ends right before stim onset
        light_baseline_win = avg_freqs["Light Avg Frequency (Hz)"].loc[
            response_start - 100 : response_start - 10
        ]
        spon_baseline_win = avg_freqs["Spontaneous Avg Frequency (Hz)"].loc[
            response_start - 100 : response_start - 10
        ]

        # gets avg baseline
        freq_stats_file = os.path.join(
            self.tables_folder, f"{cell_name}_avg_freq_stats.csv"
        )
        freq_stats = pd.read_csv(freq_stats_file, index_col=0)

        light_baseline = freq_stats.loc["Light"]["Baseline Frequency (Hz)"]
        spon_baseline = freq_stats.loc["Spontaneous"][
            "Baseline Frequency (Hz)"
        ]

        nested_dict = lambda: defaultdict(nested_dict)
        nest_pval_dict = nested_dict()

        subtracts = ["no sub", "sub avg", "sub baseline"]
        comparison_types = [
            "light vs. spon",
            "light vs. baseline",
            "spon vs. baseline",
        ]

        for comparison in comparison_types:
            if comparison == "light vs. spon":
                x = light_freq
                y = spon_freq
            elif comparison == "light vs. baseline":
                x = light_freq
                y = light_baseline_win
            elif comparison == "spon vs. baseline":
                x = spon_freq
                y = spon_baseline_win

            if x.equals(light_freq):
                x_baseline = light_baseline
            elif x.equals(spon_freq):
                x_baseline = spon_baseline

            if y.equals(light_freq):
                y_baseline = light_baseline
            if y.equals(spon_freq):
                y_baseline = spon_baseline

            for subtract_type in subtracts:

                if subtract_type == "sub avg":
                    x = x - x.mean()
                    y = y - y.mean()

                elif subtract_type == "sub baseline":
                    x = x - x_baseline
                    y = y - y_baseline

                (
                    ttest_stats,
                    ttest_pval,
                    ks_stats,
                    ks_pval,
                ) = self.run_freq_stats(x, y)

                nest_pval_dict[comparison][subtract_type][
                    "ttest stats"
                ] = ttest_stats

                nest_pval_dict[comparison][subtract_type][
                    "ttest pval"
                ] = ttest_pval
                nest_pval_dict[comparison][subtract_type][
                    "ks stats"
                ] = ks_stats
                nest_pval_dict[comparison][subtract_type]["ks pval"] = ks_pval

                freqs_used = pd.DataFrame({"x": x.values, "y": y.values})
                freqs_used.index = x.index  # add time indices back in
                nest_pval_dict[comparison][subtract_type]["freqs"] = freqs_used

        self.response_pvals = nest_pval_dict

        # plot traces to see what's going on
        colors = {"light": "#B958F2", "spon": "#8A8C89", "baseline": "#ECA238"}

        self.stats_fig = plot_response_win_comparison(
            self.dataset,
            self.cell_type,
            self.cell_name,
            self.stim_time,
            colors,
            self.response_pvals,
        )

    def run_freq_stats(self, x, y):
        """
        Runs t-test and KS test on two arrays, returning p-values. Runs paired
        t-test.
        """
        ttest_stats = pg.ttest(x, y, paired=True)
        ttest_pval = ttest_stats["p-val"][0]
        ks_stats, ks_pval = scipy.stats.ks_2samp(x, y)

        return ttest_stats, ttest_pval, ks_stats, ks_pval

    def check_light_response(self):
        """
        Uses ttest p-value of light vs. baseline, no sub comparison AND 
        whether mean trace peak exceeds 3 std of baseline to determine whether 
        cell has light-evoked response.
        """

        light_pval = self.response_pvals["light vs. baseline"]["no sub"][
            "ttest pval"
        ]

        if light_pval < 0.05:
            self.light_freq_response = True
        else:
            self.light_freq_response = False

        spon_pval = self.response_pvals["spon vs. baseline"]["no sub"][
            "ttest pval"
        ]

        if spon_pval < 0.05:
            self.spon_freq_response = True
        else:
            self.spon_freq_response = False

        # gets mean trace peak info
        file_name = f"{self.cell_name}_mean_trace_stats.csv"
        mean_trace_stats_file = os.path.join(self.tables_folder, file_name)

        mean_trace_stats = pd.read_csv(mean_trace_stats_file, index_col=0)

        self.mean_trace_response = mean_trace_stats.loc["Light"][
            "Response 4x STD"
        ]

        if self.light_freq_response is True & self.mean_trace_response is True:
            self.overall_response = True
        else:
            self.overall_response = False

    def save_light_response(self):
        """
        Saves checked frequency and mean trace responses to csv
        """
        responses_df = pd.DataFrame(
            {
                "timepoint": self.dataset,
                "cell name": self.cell_name,
                "cell type": self.cell_type,
                "light freq response": self.light_freq_response,
                "mean trace response": self.mean_trace_response,
                "spon freq response": self.spon_freq_response,
                "overall respnse": self.overall_response,
            },
            index=[0],
        )
        responses_df = responses_df.T

        file_name = f"{self.cell_name}_responses.csv"
        path = os.path.join(self.tables_folder, file_name)

        responses_df.to_csv(
            path, float_format="%8.4f", index=True, header=False
        )

    def save_stats_fig(self):
        """
        Saves the response comparisons plot with t-tests and KS-tests
        """

        html_filename = f"{self.cell_name}_comparison_pval_plots.html"
        path = os.path.join(self.figs_folder, html_filename)

        self.stats_fig.write_html(
            path, full_html=False, include_plotlyjs="cdn"
        )


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

    # compares avg freqs between light and spon condition to determine
    # whether or not cell has response
    cell.compare_avg_freqs()
    cell.check_light_response()
    cell.save_light_response()
    cell.save_stats_fig()

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
    # cell_name = "JH200313_c2"
    cell_name = "JH190905_c7"

    run_both_conditions(dataset, csvfile, cell_name)

    # file_name = "JH200303_c7_light100.ibw"

    # run_single(dataset, csvfile, file_name)

    print("Analysis for {} done".format(cell_name))
