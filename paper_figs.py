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

from scipy.stats import sem

import plotly.io as pio

pio.renderers.default = "browser"

from p2_acq_parameters import *
from p14_acq_parameters import *
import pdb
from single_test import JaneCell
from plotting import *
from aggregate_stats import *


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


# def get_cell_sweeps_dict(cell, spikes=False):
#     """
#     Gets the dict of all sweeps for a cell object
#     """
#     # 2 drops depolarized and esc AP sweeps from VC data if applicable
#     cell.drop_sweeps()

#     # 3 makes a dict for each cell, with stim condition as keys and all sweeps per stimulus as values
#     if spikes is False:
#         cell.make_sweeps_dict()
#         return cell.sweeps_dict

#     else:
#         cell.make_spikes_dict()
#         return cell.ic_sweeps_dict


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


def make_annotated_trace(dataset, csvfile, genotype, file_name, sweep_number):
    """
    Plots a single VC trace to demonstrate onset latency, peak amplitude, and
    time to peak.
    """

    cell = get_single_cell(dataset, csvfile, file_name)
    traces, annotation_values = get_single_cell_traces(
        cell, "single", sweep_number, annotate=True
    )
    axes, noaxes = plot_annotated_trace(traces, annotation_values, genotype)
    save_annotated_figs(axes, noaxes, cell, genotype)

    print("Finished saving annotated trace plots")


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
        "Control": {
            "name": "JH190828_c6_light100.ibw",
            "cell type": "Control",
            "timepoint": "p14",
            "sweep to use": 3,
        },
        "Gabazine": {
            "name": "JH190828Gabazine_c6_light100.ibw",
            "cell type": "Gabazine",
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

    return fig


if __name__ == "__main__":

    # sections_data = get_ephys_sections_intensity()
    # (
    #     sections_fig,
    #     sections_fig_data,
    #     sections_regression,
    # ) = plot_ephys_sections_intensity(sections_data)

    # # save_ephys_sections_fig(
    # #     sections_fig, sections_fig_data, sections_regression
    # # )

    # save_fig_to_png(
    #     sections_fig,
    #     legend=False,
    #     rows=1,
    #     cols=2,
    #     png_filename="ephys_sections_comparisons.png",
    # )

    # epl_fig = plot_EPL_intensity()  # should move this to paper figs
    # # save_epl_plot(epl_fig)

    # save_fig_to_png(
    #     epl_fig,
    #     legend=False,
    #     rows=1,
    #     cols=1,
    #     png_filename="EPL_intensity_plot.png",
    # )

    # p2_6wpi_counts_fig = plot_p2_6wpi_response_counts()
    # # save_p2_6wpi_counts_fig(p2_6wpi_counts_fig)

    # save_fig_to_png(
    #     p2_6wpi_counts_fig,
    #     legend=True,
    #     rows=1,
    #     cols=1,
    #     png_filename="p2_6wpi_response_plot.png",
    # )

    # # make example traces
    # make_GC_example_traces()
    # make_timepoint_example_traces()
    # make_gabazine_wash_in_traces()

    # example_PSTH_fig = get_example_cell_PSTH("p14", "JH190828_c6")

    # save_fig_to_png(
    #     example_PSTH_fig,
    #     legend=True,
    #     rows=2,
    #     cols=3,
    #     png_filename="example_PSTH_fig.png",
    # )

    prev_analysis_fig = plot_previous_analysis()
    save_fig_to_png(
        prev_analysis_fig,
        legend=True,
        rows=1,
        cols=1,
        png_filename="prev_analysis_fig.png",
    )

    pdb.set_trace()

