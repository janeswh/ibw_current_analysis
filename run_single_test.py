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

import plotly.io as pio

pio.renderers.default = "browser"

import p2_acq_parameters
import p14_acq_parameters
import pdb
from single_test import JaneCell
from file_settings import FileSettings


def get_both_conditions(dataset, csvfile, cell_name):
    """
    Should do this to avoid repeatedly running analysis
    1. get cell object for each condition
    2. save all the data to csv
    3. import data and plot
    4. save plots
    """
    file_names = [
        f"{cell_name}_light100.ibw",
        f"{cell_name}_spontaneous.ibw",
    ]

    light_cell = run_single(dataset, csvfile, file_names[0])
    spon_cell = run_single(dataset, csvfile, file_names[1])

    cell_type = light_cell.cell_type
    stim_time = light_cell.stim_time

    # should save avg frequency traces to csv so we don't have to run
    # the bayesian stuff every time

    # also save avg frequency stats
    # save event stats
    save_freqs(
        dataset,
        cell_type,
        cell_name,
        light_cell.freq,
        light_cell.avg_frequency_df,
        spon_cell.freq,
        spon_cell.avg_frequency_df,
    )

    save_event_stats(
        dataset,
        cell_type,
        cell_name,
        light_cell.event_stats,
        spon_cell.event_stats,
    )

    stats_fig = plot_event_stats(dataset, cell_name, cell_type)
    freqs_fig = plot_both_freqs(dataset, cell_name, cell_type, stim_time)

    output_html_plots(stats_fig, freqs_fig, dataset, cell_type, cell_name)

    response = check_response(dataset, cell_type, cell_name)
    if response is True:
        print("cell has response")

    pdb.set_trace()


def check_response(dataset, cell_type, cell_name):
    # determine whether cell has response with KS test
    p_value = run_KS_test(dataset, cell_type, cell_name)

    if p_value <= 0.05:
        response = True
    else:
        response = False

    return response


def save_event_stats(dataset, cell_type, cell_name, light_stats, spon_stats):

    # saves individual event stats

    # drops rise start and end columns
    cols_to_drop = ["Rise start (ms)", "Rise end (ms)", "Root time (ms)"]
    light_stats.drop(labels=cols_to_drop, axis=1, inplace=True)
    spon_stats.drop(labels=cols_to_drop, axis=1, inplace=True)

    base_path = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
    )
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_names = [
        f"{cell_name}_light_event_stats.csv",
        f"{cell_name}_spontaneous_event_stats.csv",
    ]
    dfs = [light_stats, spon_stats]

    for count, file_name in enumerate(file_names):
        path = os.path.join(base_path, file_name)
        dfs[count].to_csv(path, float_format="%8.4f", index=True)

    # saves avgs from both conditions in one csv
    avg_event_stats = pd.DataFrame(
        {"Light": light_stats.mean(), "Spontaneous": spon_stats.mean()},
        index=light_stats.columns,
    )

    avg_event_stats.drop(labels=["Sweep", "New pos"], axis=0, inplace=True)

    avg_stats_file_name = f"{cell_name}_avg_event_stats.csv"
    avg_stats_path = os.path.join(base_path, avg_stats_file_name)
    avg_event_stats.to_csv(avg_stats_path, float_format="%8.4f", index=True)


def save_freqs(
    dataset,
    cell_type,
    cell_name,
    light_raw_freq,
    light_avg_freq,
    spon_raw_freq,
    spon_avg_freq,
):

    raw_freqs_df = pd.DataFrame(
        {
            "Light Raw Frequency (Hz)": light_raw_freq.iloc[:, 0],
            "Spontaneous Raw Frequency (Hz)": spon_raw_freq.iloc[:, 0],
        },
        index=light_raw_freq.index,
    )

    avg_freqs_df = pd.DataFrame(
        {
            "Light Avg Frequency (Hz)": light_avg_freq.iloc[:, 0],
            "Spontaneous Avg Frequency (Hz)": spon_avg_freq.iloc[:, 0],
        },
        index=light_avg_freq.index,
    )

    base_path = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
    )
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_names = [
        f"{cell_name}_raw_frequency.csv",
        f"{cell_name}_avg_frequency.csv",
    ]
    dfs = [raw_freqs_df, avg_freqs_df]

    for count, file_name in enumerate(file_names):
        path = os.path.join(base_path, file_name)
        dfs[count].to_csv(path, float_format="%8.4f", index=True)


def run_KS_test(dataset, cell_type, cell_name):
    """
    Runs two-sample Kolmogorov-Smirnov test and returns p-value
    """
    freqs_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_avg_frequency.csv"
    )
    avg_freqs = pd.read_csv(freqs_file, index_col=0)

    stats, p_value = scipy.stats.ks_2samp(
        avg_freqs["Light Avg Frequency (Hz)"].values,
        avg_freqs["Spontaneous Avg Frequency (Hz)"].values,
    )

    return p_value


def plot_event_stats(dataset, cell_name, cell_type):

    light_stats_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_light_event_stats.csv"
    )
    light_stats_df = pd.read_csv(light_stats_file, index_col=0)

    spon_stats_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_spontaneous_event_stats.csv"
    )
    spon_stats_df = pd.read_csv(spon_stats_file, index_col=0)

    dfs = [light_stats_df, spon_stats_df]
    conditions = ["Light", "Spontaneous"]
    colors = ["#B958F2", "#8A8C89"]
    median_colors = ["#D49BF5", "#B9B9B9"]

    if dataset == "p2":
        stim_time = p2_acq_parameters.STIM_TIME
        post_time = p2_acq_parameters.FREQ_POST_STIM  # time to stop looking

    if dataset == "p14":
        stim_time = p14_acq_parameters.STIM_TIME
        post_time = p14_acq_parameters.FREQ_POST_STIM  # time to stop looking

    event_stats_fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Response Window Events",
            "Outside Response Window Events",
        ),
    )

    for count, df in enumerate(dfs):

        win_df_list = []

        response_win_stats = df[
            df["New pos"].between(stim_time, post_time, inclusive="both")
        ]
        outside_win_stats = df[
            ~df["New pos"].between(stim_time, post_time, inclusive="both")
        ]

        win_df_list.extend([response_win_stats, outside_win_stats])

        for win_count, condition_stats in enumerate(win_df_list):

            # event_stats_fig.update_xaxes(
            #     title_text="Response Window Events"
            #     if win_count == 0
            #     else "Outside Response Window Events",
            #     row=1,
            #     col=win_count + 1,
            # )

            # plots amplitude as histogram
            event_stats_fig.add_trace(
                go.Histogram(
                    x=condition_stats["New amplitude (pA)"],
                    marker_color=colors[count],
                    name=conditions[count],
                    legendgroup=conditions[count],
                ),
                row=1,
                col=win_count + 1,
            )

            amp_median = condition_stats["New amplitude (pA)"].median()

            event_stats_fig.add_vline(
                row=1,
                col=win_count + 1,
                x=amp_median,
                line_color=median_colors[count],
                annotation_text=f"median = {round(amp_median, 2)} pA",
                annotation_font=dict(color=median_colors[count]),
                annotation_yshift=-10 if count == 1 else 0,
                annotation_position="top right",
            )

            event_stats_fig.update_xaxes(
                title="Amplitude (pA)",
                autorange="reversed",
                row=1,
                col=win_count + 1,
            )

            # plots rise time as histogram
            event_stats_fig.add_trace(
                go.Histogram(
                    x=condition_stats["Rise time (ms)"],
                    marker_color=colors[count],
                    name=conditions[count],
                    legendgroup=conditions[count],
                ),
                row=2,
                col=win_count + 1,
            )

            rise_median = condition_stats["Rise time (ms)"].median()
            event_stats_fig.add_vline(
                row=2,
                col=win_count + 1,
                x=rise_median,
                line_color=median_colors[count],
                annotation_text=f"median = {round(rise_median, 2)} ms",
                annotation_font=dict(color=median_colors[count]),
                annotation_yshift=-10 if count == 1 else 0,
                annotation_position="top right",
            )

            event_stats_fig.update_xaxes(
                title="Rise time (ms)", row=2, col=win_count + 1
            )

            # plots tau as histogram
            event_stats_fig.add_trace(
                go.Histogram(
                    x=condition_stats["Tau (ms)"],
                    marker_color=colors[count],
                    name=conditions[count],
                    legendgroup=conditions[count],
                ),
                row=3,
                col=win_count + 1,
            )

            tau_median = condition_stats["Tau (ms)"].median()
            event_stats_fig.add_vline(
                row=3,
                col=win_count + 1,
                x=tau_median,
                line_color=median_colors[count],
                annotation_text=f"median = {round(tau_median, 2)} ms",
                annotation_font=dict(color=median_colors[count]),
                annotation_yshift=-10 if count == 1 else 0,
                annotation_position="top right",
            )

            event_stats_fig.update_xaxes(
                title="Tau (ms)", row=3, col=win_count + 1
            )

    # below is code from stack overflow to hide duplicate legends
    names = set()
    event_stats_fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    # event_stats_fig.show()

    return event_stats_fig


def plot_both_freqs(dataset, cell_name, cell_type, stim_time):

    avg_freq_csv_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_avg_frequency.csv"
    )
    avg_freqs_df = pd.read_csv(avg_freq_csv_file, index_col=0)

    raw_freq_csv_file = (
        f"{FileSettings.TABLES_FOLDER}/{dataset}/{cell_type}/{cell_name}/"
        f"{cell_name}_raw_frequency.csv"
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
        x0=stim_time,
        x1=stim_time + 100,
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
        title_text="{}, {} Frequency Comparisons".format(cell_name, cell_type),
        title_x=0.5,
    )

    # both_freqs_fig.show()

    return both_freqs_fig


def output_html_plots(stats_fig, freqs_fig, dataset, cell_type, cell_name):
    """
    Saves event stats and avg freq stats figs as one html plot
    """
    base_path = (
        f"{FileSettings.FIGURES_FOLDER}/{dataset}/{cell_type}/{cell_name}"
    )
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    html_filename = f"{cell_name}_stats_plots.html"
    path = os.path.join(base_path, html_filename)

    freqs_fig.write_html(path, full_html=False, include_plotlyjs="cdn")

    with open(path, "a") as f:
        f.write(stats_fig.to_html(full_html=False, include_plotlyjs=False))


def run_single(dataset, csvfile, file_name):
    file = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/data",
        dataset,
        file_name,
    )

    # gets sweep info for all cells
    sweep_info = pd.read_csv(csvfile, index_col=0)

    # 0 initializes JaneCell class
    cell = JaneCell(dataset, sweep_info, file, file_name)

    # 1 checks whether cell has a response before proceeding
    response = cell.check_response()

    # # 2 drops depolarized and esc AP sweeps from VC data if applicable
    # cell.drop_sweeps()

    # # 3 makes a dict for each cell, with stim condition as keys and all sweeps per stimulus as values
    # cell.make_sweeps_dict()

    # 4 runs stats on sweeps and creates a dict for each stim condition
    cell.get_mod_events()
    cell.calculate_event_stats()
    cell.calculate_mean_trace_stats()
    cell.plot_annotated_events()
    cell.save_annotated_events_plot()

    # cell.plot_events()
    cell.analyze_avg_frequency()
    cell.save_annotated_freq()

    # only save annotated histogram plot if condition == light and response
    # is true
    # should save raster plot with some form of histogram regardless
    # or, plot raster + annotated hist if response/light, else plot
    # raster + smoothed PSTH
    # if histogram decay_fits fails, then don't plot annotated

    # pdb.set_trace()

    cell.plot_mean_trace()
    # cell.save_mean_trace_plot()   # don't save, each plot is 130 mb

    # # cell.make_cell_analysis_dict()

    # # 5 calculates power curve for plotting
    # cell.make_power_curve_stats_df()

    # # 6 calculates response stats for plotting
    # cell.make_stats_df()

    # # 7 plots mean traces
    # cell.make_mean_traces_df()
    # cell.graph_response_trace()

    # # 8 makes plots for power curve and response stats if cell responds
    # if response == True:

    #     summary_plots = cell.graph_curve_stats()
    #     cell.export_stats_csv()
    # else:
    #     print("Cell doesn't have response, no response stats plotted")

    # # 9 saves combined plots as html file, exports stats as csv
    # cell.output_html_plots()

    return cell


if __name__ == "__main__":
    dataset = "p2"
    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables",
        dataset,
        csvfile_name,
    )
    cell_name = "JH200311_c1"

    get_both_conditions(dataset, csvfile, cell_name)
    # file_name = "JH200303_c7_light100.ibw"

    # run_single(dataset, csvfile, file_name)

    print("Analysis for {} done".format(cell_name))
