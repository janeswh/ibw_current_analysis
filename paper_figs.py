from unicodedata import name
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
from file_settings import FileSettings

import collections
from scipy.stats import sem

import plotly.io as pio

pio.renderers.default = "browser"

from p2_acq_parameters import *
from p14_acq_parameters import *
import pdb
from single_test import JaneCell
from plotting import *
from aggregate_stats import *
from run_datasets import get_datasets


def get_single_cell(dataset, csvfile, nwbfile_name):
    """
    Initiates the cell object for extracting traces later
    """
    nwbfile = os.path.join(
        "/home/jhuang/Documents/phd_projects/MMZ_STC_dataset/data",
        dataset,
        nwbfile_name,
    )

    # gets sweep info for all cells
    sweep_info = pd.read_csv(csvfile, index_col=0)

    # reads in NWB file
    io = NWBHDF5IO(nwbfile, "r", load_namespaces=True)
    nwbfile = io.read()

    # 0 initializes JaneCell class
    cell = JaneCell(dataset, sweep_info, nwbfile, nwbfile_name)

    return cell


def get_annotation_values(cell, selected_condition, sweep_number):
    """
    Gets the response onset, amplitude peak, and time to peak values for
    annotating example trace.
    """

    onset_time = cell.cell_analysis_dict[selected_condition][
        "Onset Times (ms)"
    ][sweep_number]

    onset_latency = cell.cell_analysis_dict[selected_condition][
        "Onset Latencies (ms)"
    ][sweep_number]

    amplitude_peak = cell.cell_analysis_dict[selected_condition][
        "Raw Peaks (pA)"
    ][sweep_number]

    time_topeak = cell.cell_analysis_dict[selected_condition][
        "Time to Peaks (ms)"
    ][sweep_number]

    return [onset_time, onset_latency, amplitude_peak, time_topeak]


def get_single_cell_traces(
    cell, traces_type="mean", sweep_number=None, annotate=False
):
    """
    Gets the normal light stim mean traces for a cell object
    """

    # 2 drops depolarized and esc AP sweeps from VC data if applicable
    cell.drop_sweeps()

    # 3 makes a dict for each cell, with stim condition as keys and all sweeps per stimulus as values
    if traces_type == "spikes":
        cell.make_spikes_dict()
    else:
        cell.make_sweeps_dict()

        # 4 runs stats on sweeps and creates a dict for each stim condition
        cell.make_cell_analysis_dict()

        # 5 calculates power curve for plotting
        cell.make_power_curve_stats_df()

        # 6 calculates response stats for plotting
        cell.make_stats_df()

    if cell.cell_name == "JH20210923_c2":
        selected_condition = "50%, 1 ms"
    else:
        selected_condition = ",".join(FileSettings.SELECTED_CONDITION)

    if annotate is True:
        annotation_values = get_annotation_values(
            cell, selected_condition, sweep_number
        )
    else:
        annotation_values = None

    if traces_type == "mean":
        cell.make_mean_traces_df()
        traces = cell.mean_trace_df

    elif traces_type == "spikes":
        # # pulls out spikes sweep
        spike_sweep = cell.extract_FI_sweep(sweep_number)
        traces = spike_sweep

    elif traces_type == "single":
        vc_sweep = cell.filtered_traces_dict[selected_condition][sweep_number]
        # vc_sweep = cell.extract_VC_sweep(selected_condition, sweep_number)
        traces = vc_sweep

    return traces, annotation_values


def make_GC_example_traces():

    files_dict = {
        "GC cell-attached": {
            "name": "JH200303GCAttached_c5_light100.ibw",
            "cell type": "GC cell-attached",
            "timepoint": "extra_sweeps",
            "sweep to use": 0,
        },
        "GC break-in": {
            "name": "JH200303GCBrokeIn_c5_light100.ibw",
            "cell type": "GC break-in",
            "timepoint": "extra_sweeps",
            "sweep to use": 0,
        },
    }
    ephys_traces_plotted = pd.DataFrame()

    dataset = "extra_sweeps"
    csvfile_name = f"{dataset}_data_notes.csv"
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, dataset, csvfile_name,)
    sweep_info = pd.read_csv(csvfile, index_col=0)

    for cell, info in files_dict.items():
        dataset = info["timepoint"]
        file = os.path.join(FileSettings.DATA_FOLDER, dataset, info["name"])

        file_sweeps = JaneCell(dataset, sweep_info, file, info["name"])
        file_sweeps.initialize_cell()
        file_traces = file_sweeps.process_traces_for_plotting()
        file_plotting_trace, plotted_trace = make_one_plot_trace(
            info["name"],
            cell_trace=file_traces[info["sweep to use"]],
            type=info["cell type"],
            inset=False,
        )
        files_dict[cell]["plotting trace"] = file_plotting_trace
        files_dict[cell]["ephys trace"] = plotted_trace

        plotted_trace = pd.DataFrame({cell: plotted_trace})
        ephys_traces_plotted = pd.concat(
            [ephys_traces_plotted, plotted_trace], axis=1
        )

    example_gc_fig = plot_example_GC_traces(files_dict)
    save_example_traces_figs(example_gc_fig, ephys_traces_plotted, "GC")


def make_timepoint_example_traces():
    # test on p2
    files_dict = {
        "p2": {
            "MC 1": {
                "name": "JH200311_c1_light100.ibw",
                "cell type": "MC",
                "timepoint": "p2",
                "sweep to use": 18,
            },
            "MC 2": {
                "name": "JH200313_c3_light100.ibw",
                "cell type": "MC",
                "timepoint": "p2",
                "sweep to use": 4,
            },
            "TC 1": {
                "name": "JH200311_c2_light100.ibw",
                "cell type": "TC",
                "timepoint": "p2",
                "sweep to use": 4,
            },
            "TC 2": {
                "name": "JH20210812_c6_light100.ibw",
                "cell type": "TC",
                "timepoint": "p2",
                "sweep to use": 1,
            },
        },
        "p14": {
            "MC 1": {
                "name": "JH190828_c6_light100.ibw",
                "cell type": "MC",
                "timepoint": "p14",
                "sweep to use": 3,
            },
            "MC 2": {
                "name": "JH191008_c5_light100.ibw",
                "cell type": "MC",
                "timepoint": "p14",
                "sweep to use": 26,
            },
            "TC 1": {
                "name": "JH191009_c3_light100.ibw",
                "cell type": "TC",
                "timepoint": "p14",
                "sweep to use": 4,
            },
            "TC 2": {
                "name": "JH191008_c4_light100.ibw",
                "cell type": "TC",
                "timepoint": "p14",
                "sweep to use": 18,
            },
        },
    }

    for timepoint, file_list in files_dict.items():
        ephys_traces_plotted = pd.DataFrame()

        dataset = timepoint
        csvfile_name = f"{dataset}_data_notes.csv"
        csvfile = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, csvfile_name,
        )
        sweep_info = pd.read_csv(csvfile, index_col=0)

        for cell, info in file_list.items():
            dataset = info["timepoint"]
            file = os.path.join(
                FileSettings.DATA_FOLDER, dataset, info["name"]
            )

            file_sweeps = JaneCell(dataset, sweep_info, file, info["name"])
            file_sweeps.initialize_cell()
            file_traces = file_sweeps.process_traces_for_plotting()
            file_plotting_trace, plotted_trace = make_one_plot_trace(
                info["name"],
                cell_trace=file_traces[info["sweep to use"]],
                type=info["cell type"],
                inset=False,
            )
            files_dict[timepoint][cell]["plotting trace"] = file_plotting_trace
            files_dict[timepoint][cell]["ephys trace"] = plotted_trace

            plotted_trace = pd.DataFrame({cell: plotted_trace})
            ephys_traces_plotted = pd.concat(
                [ephys_traces_plotted, plotted_trace], axis=1
            )

        example_fig = plot_example_cell_type_traces(
            files_dict[timepoint], timepoint
        )
        save_example_traces_figs(example_fig, ephys_traces_plotted, timepoint)


def make_gabazine_wash_in_traces():

    files_dict = {
        "MC Control": {
            "name": "JH190828_c6_light100.ibw",
            "cell type": "MC Control",
            "timepoint": "p14",
            "sweep to use": 3,
        },
        "MC Gabazine": {
            "name": "JH190828Gabazine_c6_light100.ibw",
            "cell type": "MC Gabazine",
            "timepoint": "extra_sweeps",
            "sweep to use": 26,
        },
    }
    ephys_traces_plotted = pd.DataFrame()

    for cell, info in files_dict.items():
        dataset = info["timepoint"]
        csvfile_name = f"{dataset}_data_notes.csv"
        csvfile = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, csvfile_name,
        )
        sweep_info = pd.read_csv(csvfile, index_col=0)
        file = os.path.join(FileSettings.DATA_FOLDER, dataset, info["name"])

        file_sweeps = JaneCell(dataset, sweep_info, file, info["name"])
        file_sweeps.initialize_cell()
        file_traces = file_sweeps.process_traces_for_plotting()
        file_plotting_trace, plotted_trace = make_one_plot_trace(
            info["name"],
            cell_trace=file_traces[info["sweep to use"]],
            type=info["cell type"],
            inset=False,
        )
        files_dict[cell]["plotting trace"] = file_plotting_trace
        files_dict[cell]["ephys trace"] = plotted_trace

        plotted_trace = pd.DataFrame({cell: plotted_trace})
        ephys_traces_plotted = pd.concat(
            [ephys_traces_plotted, plotted_trace], axis=1
        )

    drug_wash_fig = plot_gabazine_wash_traces(files_dict)
    save_example_traces_figs(
        drug_wash_fig, ephys_traces_plotted, "gabazine_wash"
    )


def make_example_huge_trace():

    files_dict = {
        "MC": {
            "name": "JH20210824_c4_light100.ibw",
            "cell type": "MC Control",
            "timepoint": "p2_6wpi",
            "sweep to use": 0,
        }
    }

    ephys_traces_plotted = pd.DataFrame()

    for cell, info in files_dict.items():
        dataset = info["timepoint"]
        csvfile_name = f"{dataset}_data_notes.csv"
        csvfile = os.path.join(
            FileSettings.TABLES_FOLDER, dataset, csvfile_name,
        )
        sweep_info = pd.read_csv(csvfile, index_col=0)
        file = os.path.join(FileSettings.DATA_FOLDER, dataset, info["name"])

        file_sweeps = JaneCell(dataset, sweep_info, file, info["name"])
        file_sweeps.initialize_cell()
        file_traces = file_sweeps.process_traces_for_plotting()
        file_plotting_trace, plotted_trace = make_one_plot_trace(
            info["name"],
            cell_trace=file_traces[info["sweep to use"]],
            type=info["cell type"],
            inset=False,
        )
        files_dict[cell]["plotting trace"] = file_plotting_trace
        files_dict[cell]["ephys trace"] = plotted_trace

        plotted_trace = pd.DataFrame({cell: plotted_trace})
        ephys_traces_plotted = pd.concat(
            [ephys_traces_plotted, plotted_trace], axis=1
        )

    fig = plot_single_trace(files_dict, timepoint="p2")
    save_example_traces_figs(fig, ephys_traces_plotted, "p2_6wpi_huge")


def get_example_cell_PSTH(dataset, cell_name):
    """
    Gets the raster plot and PSTH plot of one cell for showing an example.
    """
    csvfile_name = f"{dataset}_data_notes.csv"
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, dataset, csvfile_name,)

    file_name = f"{cell_name}_light100.ibw"
    sweep_info = pd.read_csv(csvfile, index_col=0)
    file = os.path.join(FileSettings.DATA_FOLDER, dataset, file_name)

    example_cell = JaneCell(dataset, sweep_info, file, file_name)
    example_cell.check_response()
    example_cell.get_mod_events()
    example_cell.calculate_event_stats()
    example_cell.calculate_mean_trace_stats()
    example_cell.analyze_avg_frequency()

    fig = example_cell.annotated_freq_fig

    # gives x-axis label room to breathe
    fig.for_each_annotation(lambda a: a.update(yshift=-50))

    save_fig_to_png(
        fig,
        legend=True,
        rows=2,
        cols=3,
        png_filename="test.png",
        extra_bottom=True,
    )

    return fig


def get_correlations(data_type):
    if data_type == "event":
        csvfile_name = "cell_comparison_medians_data.csv"
    elif data_type == "frequency":
        csvfile_name = "freq_stats_data.csv"

    csvfile = os.path.join(
        FileSettings.TABLES_FOLDER, "datasets_summaries_data", csvfile_name,
    )

    df = pd.read_csv(csvfile, index_col=0, header=0)
    all_corrs = pd.DataFrame()
    figs_list = []

    if data_type == "event":
        correlations = [
            ["Adjusted amplitude (pA)", "Rise time (ms)"],
            ["Adjusted amplitude (pA)", "Tau (ms)"],
            ["Rise time (ms)", "Tau (ms)"],
        ]

        fig_names = ["amp_rise", "amp_tau", "rise_tau"]

    if data_type == "frequency":
        correlations = [["Baseline-sub Peak Freq (Hz)", "Rise Time (ms)"]]
        fig_names = ["peak_freq_rise"]

    for corr_pair in correlations:
        corr_fig, corr_stats = plot_correlations(
            df, data_type, corr_pair[0], corr_pair[1]
        )

        figs_list.append(corr_fig)
        all_corrs = pd.concat([all_corrs, corr_stats])

    save_corr_fig(figs_list, all_corrs, data_type)

    for count, fig in enumerate(figs_list):
        save_fig_to_png(
            fig,
            legend=True,
            rows=2,
            cols=2,
            png_filename=f"{data_type}_{fig_names[count]}_corr.png",
            extra_bottom=True,
        )


def plot_misc_data():

    # gets ephys intensity data
    sections_data = get_ephys_sections_intensity()

    (
        sections_fig,
        sections_fig_data,
        sections_corr,
    ) = plot_ephys_sections_intensity(sections_data)

    save_ephys_sections_fig(sections_fig, sections_fig_data, sections_corr)

    save_fig_to_png(
        sections_fig,
        legend=False,
        rows=1,
        cols=2,
        png_filename="combined_ephys_sections_comparisons.png",
        extra_bottom=True,
    )

    # gets ephys intensity data but separated out timepoints
    (
        timepoint_sections_fig,
        timepoint_sections_fig_data,
        timepoint_sections_corr,
    ) = plot_ephys_sections_intensity_timepoint(sections_data)

    save_ephys_sections_fig(
        timepoint_sections_fig,
        timepoint_sections_fig_data,
        timepoint_sections_corr,
        timepoint=True,
    )

    save_fig_to_png(
        timepoint_sections_fig,
        legend=True,
        rows=2,
        cols=2,
        png_filename="timepoint_ephys_sections_comparisons.png",
        extra_bottom=True,
    )

    epl_fig = plot_EPL_intensity()  # should move this to paper figs
    save_epl_plot(epl_fig)

    save_fig_to_png(
        epl_fig,
        legend=False,
        rows=1,
        cols=1,
        png_filename="EPL_intensity_plot.png",
    )

    p2_6wpi_counts_fig = plot_p2_6wpi_response_counts()
    save_p2_6wpi_counts_fig(p2_6wpi_counts_fig)

    save_fig_to_png(
        p2_6wpi_counts_fig,
        legend=True,
        rows=1,
        cols=1,
        png_filename="p2_6wpi_response_plot.png",
    )

    # plots previously analyzed mean trace peak data
    prev_analysis_fig = plot_previous_analysis()
    save_fig_to_png(
        prev_analysis_fig,
        legend=True,
        rows=1,
        cols=1,
        png_filename="prev_analysis_fig.png",
    )

    example_PSTH_fig = get_example_cell_PSTH("p14", "JH190828_c6")
    save_fig_to_png(
        example_PSTH_fig,
        legend=True,
        rows=2,
        cols=3,
        png_filename="example_PSTH_fig.png",
        extra_bottom=True,
    )


def get_avg_frequencies():
    """
    Gets the average frequencies for each cell for plotting
    """
    dataset_list = get_datasets()

    avg_freqs_dict = collections.defaultdict(dict)
    for dataset in dataset_list:
        cell_types = [
            path
            for path in os.listdir(
                os.path.join(FileSettings.TABLES_FOLDER, dataset)
            )
            if os.path.isdir(
                os.path.join(FileSettings.TABLES_FOLDER, dataset, path)
            )
        ]

        cell_responses = pd.read_csv(
            os.path.join(
                FileSettings.TABLES_FOLDER,
                dataset,
                f"{dataset}_response_cells_list.csv",
            ),
            index_col=0,
        )

        responding_cells = cell_responses.loc[
            cell_responses["datanotes eye response"] == True
        ]

        for cell_type in cell_types:
            freqs_df = pd.DataFrame()
            cells_list = os.listdir(
                os.path.join(FileSettings.TABLES_FOLDER, dataset, cell_type)
            )

            cells_list = responding_cells.loc[
                responding_cells["cell type"] == cell_type
            ]

            for cell_name in cells_list["cell_name"]:
                freq_file = f"{cell_name}_avg_frequency.csv"
                csvfile = os.path.join(
                    FileSettings.TABLES_FOLDER,
                    dataset,
                    cell_type,
                    cell_name,
                    freq_file,
                )
                cell_df = pd.read_csv(csvfile, index_col=0)
                cell_df.drop(
                    labels="Spontaneous Avg Frequency (Hz)",
                    axis=1,
                    inplace=True,
                )
                cell_df = cell_df.rename(
                    columns={"Light Avg Frequency (Hz)": cell_name}
                )
                freqs_df = pd.concat([freqs_df, cell_df], axis=1)

            avg_freqs_dict[dataset][cell_type] = freqs_df

    return avg_freqs_dict


def make_avg_freq_traces():
    freqs = get_avg_frequencies()

    for dataset in list(freqs.keys()):
        fig = plot_avg_freq(
            freqs[dataset]["MC"], freqs[dataset]["TC"], dataset
        )
        save_fig_to_png(
            fig,
            legend=True,
            rows=2,
            cols=1,
            png_filename=f"{dataset}_frequency_traces.png",
        )


if __name__ == "__main__":
    make_avg_freq_traces()

    pdb.set_trace()

    plot_misc_data()

    get_correlations(data_type="event")
    get_correlations(data_type="frequency")

    make_example_huge_trace()

    # make example traces
    make_GC_example_traces()
    make_timepoint_example_traces()
    make_gabazine_wash_in_traces()

