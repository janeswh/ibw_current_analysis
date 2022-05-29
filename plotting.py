from locale import D_FMT
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.default_scale = 5
pio.kaleido.scope.default_format = "png"
from scipy.stats import sem
from collections import defaultdict
import plotly.io as pio
from file_settings import FileSettings

pio.renderers.default = "browser"
import pdb


def plot_averages(dataset, genotype, threshold, averages_df):

    # plotting
    durations = averages_df["Light Duration"].unique()
    intensities = averages_df["Light Intensity"].unique()
    color_dict = {
        " 2 ms": "#7D1935",
        " 1 ms": "#B42B51",
        " 0.25 ms": "#E63E6D",
        " 0.01 ms": "#F892B9",
    }
    summary_stats_fig = make_subplots(
        rows=3, cols=2, x_title="Light Intensity (%)"
    )

    for count, duration in enumerate(durations):

        y_df = averages_df.loc[averages_df["Light Duration"] == duration]
        # pdb.set_trace()
        x_intensity = y_df["Light Intensity"]

        # mean trace peak amplitude
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Mean Trace Peak (pA)"],
                # if len(y_df) > 1
                # else averages_df.loc[
                #     averages_df["Light Duration"] == duration,
                #     ["Mean Trace Peak (pA)"],
                # ],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=1,
            col=1,
        )

        # Mean Onset Latency (ms)
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Mean Onset Latency (ms)"],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=1,
            col=2,
        )

        # onset jitter
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Onset Jitter"],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=2,
            col=1,
        )

        # mean trace onset latency
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Mean Trace Onset Latency (ms)"],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=2,
            col=2,
        )

        # mean time to peak
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Mean Time to Peak (ms)"],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=3,
            col=1,
        )

        # mean trace time to peak
        summary_stats_fig.add_trace(
            go.Box(
                x=x_intensity,
                y=y_df["Mean Trace Time to Peak (ms)"],
                name=duration,
                line=dict(color=color_dict[duration]),
                legendgroup=duration,
            ),
            row=3,
            col=2,
        )

        # Update xaxis properties
        # summary_stats_fig.update_xaxes(autorange="reversed")
        # this defines the intensities order for x-axes
        summary_stats_fig.update_xaxes(
            categoryorder="array", categoryarray=np.flip(intensities)
        )

        # Update yaxis properties
        summary_stats_fig.update_yaxes(
            title_text="Mean Response Amplitude (pA)",
            row=1,
            col=1,
            autorange="reversed",
        )
        summary_stats_fig.update_yaxes(
            title_text="Mean Onset Latency (ms)", row=1, col=2
        )
        summary_stats_fig.update_yaxes(
            title_text="Mean Onset Jitter", row=2, col=1
        )
        summary_stats_fig.update_yaxes(
            title_text="Mean Trace Onset Latency (ms)", row=2, col=2
        )
        summary_stats_fig.update_yaxes(
            title_text="Mean Time to Peak (ms)", row=3, col=1
        )
        summary_stats_fig.update_yaxes(
            title_text="Mean Trace Time to Peak (ms)", row=3, col=2
        )

    summary_stats_fig.update_layout(
        # yaxis_title='Onset Latency (ms)',
        boxmode="group"  # group together boxes of the different traces for each value of x
    )

    # below is code from stack overflow to hide duplicate legends
    names = set()
    summary_stats_fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    summary_stats_fig.update_layout(
        legend_title_text="Light Duration",
        title_text=(
            dataset
            + " "
            + genotype
            + " summary values, "
            + str(threshold)
            + " mean onset latency threshold"
        ),
        title_x=0.5,
    )

    # summary_stats_fig.show()

    return summary_stats_fig


def save_summary_stats_fig(genotype, threshold, fig_folder, fig):

    html_filename = "{}_{}_threshold_summary_avgs.html".format(
        genotype, threshold
    )
    path = os.path.join(fig_folder, html_filename)

    fig.write_html(path, full_html=False, include_plotlyjs="cdn")


def plot_selected_averages(threshold, selected_avgs):

    genotype_color = {"OMP": "#ff9300", "Gg8": "#7a81ff"}

    selected_summary_fig = make_subplots(rows=3, cols=2, x_title="Dataset")

    genotypes = selected_avgs["Genotype"].unique()

    # pdb.set_trace()
    for genotype in genotypes:

        x_datasets = selected_avgs.loc[selected_avgs["Genotype"] == genotype][
            "Dataset"
        ]

        # mean trace peak amplitude
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Mean Trace Peak (pA)"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=1,
            col=1,
        )

        # Mean Onset Latency (ms)
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Mean Onset Latency (ms)"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=1,
            col=2,
        )

        # onset jitter
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Onset Jitter"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=2,
            col=1,
        )

        # mean trace onset latency
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Mean Trace Onset Latency (ms)"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=2,
            col=2,
        )

        # mean time to peak
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Mean Time to Peak (ms)"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=3,
            col=1,
        )

        # mean trace time to peak
        selected_summary_fig.add_trace(
            go.Box(
                x=x_datasets,
                y=selected_avgs.loc[selected_avgs["Genotype"] == genotype][
                    "Mean Trace Time to Peak (ms)"
                ].squeeze(),
                name=genotype,
                line=dict(color=genotype_color[genotype]),
                legendgroup=genotype,
            ),
            row=3,
            col=2,
        )

        # Update xaxis properties
        # selected_summary_fig.update_xaxes(autorange="reversed")
        # this defines the dataset order for x-axes
        dataset_order = [
            "non-injected",
            "3dpi",
            "5dpi",
            "dox_3dpi",
            "dox_4dpi",
            "dox_5dpi",
        ]

        selected_summary_fig.update_xaxes(
            categoryorder="array", categoryarray=dataset_order
        )

        # Update yaxis properties
        selected_summary_fig.update_yaxes(
            title_text="Mean Response Amplitude (pA)",
            row=1,
            col=1,
            autorange="reversed",
        )
        selected_summary_fig.update_yaxes(
            title_text="Mean Onset Latency (ms)", row=1, col=2
        )
        selected_summary_fig.update_yaxes(
            title_text="Mean Onset Jitter", row=2, col=1
        )
        selected_summary_fig.update_yaxes(
            title_text="Mean Trace Onset Latency (ms)", row=2, col=2
        )
        selected_summary_fig.update_yaxes(
            title_text="Mean Time to Peak (ms)", row=3, col=1
        )
        selected_summary_fig.update_yaxes(
            title_text="Mean Trace Time to Peak (ms)", row=3, col=2
        )

    selected_summary_fig.update_layout(
        # yaxis_title='Onset Latency (ms)',
        boxmode="group"  # group together boxes of the different traces for each value of x
    )

    # below is code from stack overflow to hide duplicate legends
    names = set()
    selected_summary_fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    selected_summary_fig.update_layout(
        legend_title_text="Genotype",
        title_text=(
            "OMP vs. Gg8, {} ms onset latency threshold".format(threshold)
        ),
        title_x=0.5,
    )

    # selected_summary_fig.show()
    return selected_summary_fig


def save_selected_summary_fig(threshold, selected_summary_fig):

    html_filename = "{}_ms_threshold_datasets_summary.html".format(threshold)
    path = os.path.join(FileSettings.FIGURES_FOLDER, html_filename)

    selected_summary_fig.write_html(
        path, full_html=False, include_plotlyjs="cdn"
    )


def plot_response_counts(counts_dict):
    # response/no response is a trace

    dataset_order = {"p2": 1, "p2_4wpi": 2, "p2_6wpi": 3, "p14": 4}

    response_colors = {"no response": "#A7BBC7", "response": "#293B5F"}

    response_counts_fig = make_subplots(
        rows=1,
        cols=len(counts_dict.keys()),
        x_title="Timepoint",
        y_title="Number of Cells",
        shared_yaxes=True
        # subplot_titles=dataset_list,
    )

    for timepoint in counts_dict.keys():
        # response_csv_name = f"{timepoint}_response_counts.csv"
        # csv_file = os.path.join(
        #     FileSettings.TABLES_FOLDER, timepoint, response_csv_name
        # )

        # response_df = pd.read_csv(csv_file, header=0, index_col=0)

        for cell_type in counts_dict[timepoint].keys():
            for response_type in counts_dict[timepoint][cell_type].keys():

                response_counts_fig.add_trace(
                    go.Bar(
                        x=[cell_type],
                        y=[counts_dict[timepoint][cell_type][response_type]],
                        name=response_type,
                        marker_color=response_colors[response_type],
                        legendgroup=response_type,
                    ),
                    row=1,
                    col=dataset_order[timepoint],
                )
                response_counts_fig.update_xaxes(
                    title_text=timepoint, row=1, col=dataset_order[timepoint]
                )

    response_counts_fig.update_layout(barmode="stack")

    # below is code from stack overflow to hide duplicate legends
    names = set()
    response_counts_fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    response_counts_fig.update_layout(
        legend_title_text="Cell Responses",
        title_text=("Light-evoked Cell Responses by Onset Latency Cut-off"),
        title_x=0.5,
    )
    # response_counts_fig.show()

    return response_counts_fig


def save_response_counts_fig(response_counts_fig):

    html_filename = "all_response_counts.html"
    path = os.path.join(FileSettings.FIGURES_FOLDER, html_filename)

    response_counts_fig.write_html(
        path, full_html=False, include_plotlyjs="cdn"
    )


def plot_mean_trace_stats(mean_trace_dict):
    """
    Plots the mean trace stats for responding cells in the light condition
    """
    dataset_order = {"p2": 1, "p2_4wpi": 2, "p2_6wpi": 3, "p14": 4}
    cell_type_line_colors = {"MC": "#609a00", "TC": "#388bf7"}
    cell_type_bar_colors = {"MC": "#CEEE98", "TC": "#ACCEFA"}

    datasets = mean_trace_dict.keys()
    measures_list = ["Mean Trace Peak (pA)", "Mean Trace Peak Time (ms)"]

    mean_trace_stats_fig = make_subplots(
        rows=1, cols=len(measures_list), shared_xaxes=True, x_title="Timepoint"
    )
    all_stats = pd.DataFrame()
    all_means = pd.DataFrame()
    all_sems = pd.DataFrame()

    for timepoint in datasets:
        for cell_type_ct, cell_type in enumerate(
            mean_trace_dict[timepoint].keys()
        ):

            df = mean_trace_dict[timepoint][cell_type]["mean trace stats"]

            means = pd.DataFrame(df.mean()).T
            sem = pd.DataFrame(df.sem()).T

            means.insert(0, "Dataset", timepoint)
            means.insert(0, "Cell Type", cell_type)

            sem.insert(0, "Dataset", timepoint)
            sem.insert(0, "Cell Type", cell_type)

            all_stats = pd.concat([all_stats, df])
            all_means = pd.concat([all_means, means])
            all_sems = pd.concat([all_sems, sem])

    for cell_type_ct, cell_type in enumerate(
        mean_trace_dict[timepoint].keys()
    ):
        cell_stats_df = all_stats.loc[all_stats["Cell Type"] == cell_type]
        cell_mean_df = all_means.loc[all_means["Cell Type"] == cell_type]
        cell_sem_df = all_sems.loc[all_sems["Cell Type"] == cell_type]

        for measure_ct, measure in enumerate(measures_list):

            mean_trace_stats_fig.add_trace(
                go.Box(
                    x=cell_stats_df["Dataset"],
                    y=cell_stats_df[measure],
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints="all",
                    pointpos=0,
                    marker_color=cell_type_bar_colors[cell_type],
                    marker=dict(
                        line=dict(
                            color=cell_type_line_colors[cell_type], width=1
                        ),
                    ),
                    name=f"{cell_type} individual cells",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            # tries bar plot instead, plots mean of median with sem
            mean_trace_stats_fig.add_trace(
                go.Bar(
                    x=cell_stats_df["Dataset"].unique(),
                    y=cell_mean_df[measure],
                    error_y=dict(
                        type="data",
                        array=cell_sem_df[measure],
                        color=cell_type_line_colors[cell_type],
                        thickness=1,
                        visible=True,
                    ),
                    marker_line_color=cell_type_line_colors[cell_type],
                    marker_color=cell_type_bar_colors[cell_type],
                    # marker=dict(markercolor=cell_type_colors[cell_type]),
                    name=f"{cell_type} averages",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            if measure == "Mean Trace Peak (pA)":
                mean_trace_stats_fig.update_yaxes(
                    autorange="reversed", row=1, col=measure_ct + 1,
                )

            #  below is code from stack overflow to hide duplicate legends
            names = set()
            mean_trace_stats_fig.for_each_trace(
                lambda trace: trace.update(showlegend=False)
                if (trace.name in names)
                else names.add(trace.name)
            )
            mean_trace_stats_fig.update_yaxes(
                title_text=measure, row=1, col=measure_ct + 1,
            )
            mean_trace_stats_fig.update_xaxes(
                categoryorder="array",
                categoryarray=list(dataset_order.keys()),
                row=1,
                col=measure_ct + 1,
            )

    mean_trace_stats_fig.update_layout(
        boxmode="group", title_text="MC vs. TC Mean Trace", title_x=0.5,
    )

    # mean_trace_stats_fig.show()

    return mean_trace_stats_fig


def plot_freq_stats(dataset_freq_stats):
    """
    Plots average frequency kinetic properties for responding cells in the 
    light condition
    """
    dataset_order = {"p2": 1, "p2_4wpi": 2, "p2_6wpi": 3, "p14": 4}
    cell_type_line_colors = {"MC": "#609a00", "TC": "#388bf7"}
    cell_type_bar_colors = {"MC": "#CEEE98", "TC": "#ACCEFA"}

    datasets = dataset_freq_stats.keys()
    measures_list = [
        "Baseline-sub Peak Freq (Hz)",
        "Time to Peak Frequency (ms)",
        "Baseline Frequency (Hz)",
        "Rise Time (ms)",
        "Decay tau",
    ]

    freq_stats_fig = make_subplots(
        rows=1, cols=len(measures_list), shared_xaxes=True, x_title="Timepoint"
    )
    all_stats = pd.DataFrame()
    all_means = pd.DataFrame()
    all_sems = pd.DataFrame()

    for timepoint in datasets:
        for cell_type_ct, cell_type in enumerate(
            dataset_freq_stats[timepoint].keys()
        ):
            df = dataset_freq_stats[timepoint][cell_type]["avg freq stats"]

            means = pd.DataFrame(df.mean()).T
            sem = pd.DataFrame(df.sem()).T

            means.insert(0, "Dataset", timepoint)
            means.insert(0, "Cell Type", cell_type)

            sem.insert(0, "Dataset", timepoint)
            sem.insert(0, "Cell Type", cell_type)

            all_stats = pd.concat([all_stats, df])
            all_means = pd.concat([all_means, means])
            all_sems = pd.concat([all_sems, sem])

    for cell_type_ct, cell_type in enumerate(
        dataset_freq_stats[timepoint].keys()
    ):
        cell_stats_df = all_stats.loc[all_stats["Cell Type"] == cell_type]
        cell_mean_df = all_means.loc[all_means["Cell Type"] == cell_type]
        cell_sem_df = all_sems.loc[all_sems["Cell Type"] == cell_type]

        for measure_ct, measure in enumerate(measures_list):

            freq_stats_fig.add_trace(
                go.Box(
                    x=cell_stats_df["Dataset"],
                    y=cell_stats_df[measure],
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints="all",
                    pointpos=0,
                    marker_color=cell_type_bar_colors[cell_type],
                    marker=dict(
                        line=dict(
                            color=cell_type_line_colors[cell_type], width=1
                        ),
                    ),
                    name=f"{cell_type} individual cells",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            # tries bar plot instead, plots mean of median with sem
            freq_stats_fig.add_trace(
                go.Bar(
                    x=cell_stats_df["Dataset"].unique(),
                    y=cell_mean_df[measure],
                    error_y=dict(
                        type="data",
                        array=cell_sem_df[measure],
                        color=cell_type_line_colors[cell_type],
                        thickness=1,
                        visible=True,
                    ),
                    marker_line_color=cell_type_line_colors[cell_type],
                    marker_color=cell_type_bar_colors[cell_type],
                    # marker=dict(markercolor=cell_type_colors[cell_type]),
                    name=f"{cell_type} averages",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            if measure == "Mean Trace Peak (pA)":
                freq_stats_fig.update_yaxes(
                    autorange="reversed", row=1, col=measure_ct + 1,
                )

            #  below is code from stack overflow to hide duplicate legends
            names = set()
            freq_stats_fig.for_each_trace(
                lambda trace: trace.update(showlegend=False)
                if (trace.name in names)
                else names.add(trace.name)
            )
            freq_stats_fig.update_yaxes(
                title_text=measure, row=1, col=measure_ct + 1,
            )
            freq_stats_fig.update_xaxes(
                categoryorder="array",
                categoryarray=list(dataset_order.keys()),
                row=1,
                col=measure_ct + 1,
            )

    freq_stats_fig.update_layout(
        boxmode="group",
        title_text="MC vs. TC Avg Frequency Stats",
        title_x=0.5,
    )

    # freq_stats_fig.show()

    return freq_stats_fig


def plot_windowed_median_event_stats(median_dict):

    dataset_order = {"p2": 1, "p2_4wpi": 2, "p2_6wpi": 3, "p14": 4}
    cell_type_line_colors = {"MC": "#609a00", "TC": "#388bf7"}
    cell_type_bar_colors = {"MC": "#CEEE98", "TC": "#ACCEFA"}

    datasets = median_dict.keys()
    measures_list = ["Adjusted amplitude (pA)", "Rise time (ms)", "Tau (ms)"]
    median_fig = make_subplots(
        rows=len(measures_list),
        cols=len(datasets),
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for timepoint in datasets:
        for cell_type_ct, cell_type in enumerate(
            median_dict[timepoint].keys()
        ):
            df = median_dict[timepoint][cell_type]
            win_con_labels = {
                ("Light", "response win"): "Light response window",
                ("Spontaneous", "response win"): "Spontaneous response window",
                ("Light", "outside win"): "Light outside window",
                ("Spontaneous", "outside win"): "Spontaneous outside window",
            }

            all_medians = pd.DataFrame()
            all_means = pd.DataFrame()
            all_sems = pd.DataFrame()

            for win in df.keys():
                for condition in df[win]["Condition"].unique():
                    label = win_con_labels[(condition, win)]

                    temp_df = (
                        df[win].loc[df[win]["Condition"] == condition].copy()
                    )
                    temp_df.insert(0, "window condition", label)

                    means = pd.DataFrame(temp_df.mean()).T
                    means.index = [label]

                    sem = pd.DataFrame(temp_df.sem()).T
                    sem.index = [label]

                    all_medians = pd.concat([all_medians, temp_df])
                    all_means = pd.concat([all_means, means])
                    all_sems = pd.concat([all_sems, sem])

            for measure_ct, measure in enumerate(measures_list):
                median_fig.add_trace(
                    go.Box(
                        x=all_medians["window condition"],
                        y=all_medians[measure],
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(0,0,0,0)",
                        boxpoints="all",
                        pointpos=0,
                        marker_color=cell_type_bar_colors[cell_type],
                        marker=dict(
                            line=dict(
                                color=cell_type_line_colors[cell_type], width=1
                            ),
                        ),
                        name=f"{cell_type} medians",
                        legendgroup=cell_type,
                        offsetgroup=dataset_order[timepoint] + cell_type_ct,
                    ),
                    row=measure_ct + 1,
                    col=dataset_order[timepoint],
                )

                # tries bar plot instead, plots mean of median with sem
                median_fig.add_trace(
                    go.Bar(
                        x=all_medians["window condition"].unique(),
                        y=all_means[measure],
                        error_y=dict(
                            type="data",
                            array=all_sems[measure],
                            color=cell_type_line_colors[cell_type],
                            thickness=1,
                            visible=True,
                        ),
                        marker_line_color=cell_type_line_colors[cell_type],
                        marker_color=cell_type_bar_colors[cell_type],
                        # marker=dict(markercolor=cell_type_colors[cell_type]),
                        name=f"{cell_type} averages",
                        legendgroup=cell_type,
                        offsetgroup=dataset_order[timepoint] + cell_type_ct,
                    ),
                    row=measure_ct + 1,
                    col=dataset_order[timepoint],
                )

                if measure == "Adjusted amplitude (pA)":
                    median_fig.update_yaxes(
                        autorange="reversed",
                        row=measure_ct + 1,
                        col=dataset_order[timepoint],
                    )

                #  below is code from stack overflow to hide duplicate legends
                names = set()
                median_fig.for_each_trace(
                    lambda trace: trace.update(showlegend=False)
                    if (trace.name in names)
                    else names.add(trace.name)
                )
                median_fig.update_yaxes(
                    title_text=measure, row=measure_ct + 1, col=1
                )

            median_fig.update_xaxes(
                title_text=timepoint,
                row=len(measures_list),
                col=dataset_order[timepoint],
            )

    median_fig.update_layout(
        boxmode="group",
        title_text="Median event kinetics by response window",
        title_x=0.5,
    )
    # median_fig.show()

    return median_fig


def save_median_events_fig(windowed_medians_fig, cell_comparisons):
    """
    Saves all median event figs into one html file.
    """
    html_filename = "windowed_event_medians.html"
    path = os.path.join(FileSettings.FIGURES_FOLDER, html_filename)

    windowed_medians_fig.write_html(
        path, full_html=False, include_plotlyjs="cdn"
    )

    with open(path, "a") as f:
        f.write(
            cell_comparisons.to_html(full_html=False, include_plotlyjs=False)
        )


def save_freq_mean_trace_figs(mean_trace_fig, freq_stats_fig):
    """
    Saves mean trace stats and avg freq stats fig into one html file.
    """
    html_filename = "mean_trace_freq_stats.html"
    path = os.path.join(FileSettings.FIGURES_FOLDER, html_filename)

    mean_trace_fig.write_html(path, full_html=False, include_plotlyjs="cdn")

    with open(path, "a") as f:
        f.write(
            freq_stats_fig.to_html(full_html=False, include_plotlyjs=False)
        )


def plot_cell_type_event_comparisons(median_dict):
    """
    Compares event kinetics between MCs and TCs, for light-response windows only
    """

    dataset_order = {"p2": 1, "p2_4wpi": 2, "p2_6wpi": 3, "p14": 4}
    cell_type_line_colors = {"MC": "#609a00", "TC": "#388bf7"}
    cell_type_bar_colors = {"MC": "#CEEE98", "TC": "#ACCEFA"}

    datasets = median_dict.keys()
    measures_list = ["Adjusted amplitude (pA)", "Rise time (ms)", "Tau (ms)"]

    event_comparisons_fig = make_subplots(
        rows=1, cols=len(measures_list), shared_xaxes=True, x_title="Timepoint"
    )
    all_medians = pd.DataFrame()
    all_means = pd.DataFrame()
    all_sems = pd.DataFrame()

    for timepoint in datasets:
        for cell_type_ct, cell_type in enumerate(
            median_dict[timepoint].keys()
        ):
            df = median_dict[timepoint][cell_type]["response win"]
            temp_df = df.loc[df["Condition"] == "Light"]
            means = pd.DataFrame(temp_df.mean()).T
            sem = pd.DataFrame(temp_df.sem()).T

            means.insert(0, "Dataset", timepoint)
            means.insert(0, "Cell Type", cell_type)

            sem.insert(0, "Dataset", timepoint)
            sem.insert(0, "Cell Type", cell_type)

            all_medians = pd.concat([all_medians, temp_df])
            all_means = pd.concat([all_means, means])
            all_sems = pd.concat([all_sems, sem])

    for cell_type_ct, cell_type in enumerate(median_dict[timepoint].keys()):
        cell_median_df = all_medians.loc[all_medians["Cell Type"] == cell_type]
        cell_mean_df = all_means.loc[all_means["Cell Type"] == cell_type]
        cell_sem_df = all_sems.loc[all_sems["Cell Type"] == cell_type]

        for measure_ct, measure in enumerate(measures_list):

            event_comparisons_fig.add_trace(
                go.Box(
                    x=cell_median_df["Dataset"],
                    y=cell_median_df[measure],
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints="all",
                    pointpos=0,
                    marker_color=cell_type_bar_colors[cell_type],
                    marker=dict(
                        line=dict(
                            color=cell_type_line_colors[cell_type], width=1
                        ),
                    ),
                    name=f"{cell_type} medians",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            # tries bar plot instead, plots mean of median with sem
            event_comparisons_fig.add_trace(
                go.Bar(
                    x=cell_median_df["Dataset"].unique(),
                    y=cell_mean_df[measure],
                    error_y=dict(
                        type="data",
                        array=cell_sem_df[measure],
                        color=cell_type_line_colors[cell_type],
                        thickness=1,
                        visible=True,
                    ),
                    marker_line_color=cell_type_line_colors[cell_type],
                    marker_color=cell_type_bar_colors[cell_type],
                    # marker=dict(markercolor=cell_type_colors[cell_type]),
                    name=f"{cell_type} averages",
                    legendgroup=cell_type,
                    offsetgroup=dataset_order[timepoint] + cell_type_ct,
                ),
                row=1,
                col=measure_ct + 1,
            )

            if measure == "Adjusted amplitude (pA)":
                event_comparisons_fig.update_yaxes(
                    autorange="reversed", row=1, col=measure_ct + 1,
                )

            #  below is code from stack overflow to hide duplicate legends
            names = set()
            event_comparisons_fig.for_each_trace(
                lambda trace: trace.update(showlegend=False)
                if (trace.name in names)
                else names.add(trace.name)
            )
            event_comparisons_fig.update_yaxes(
                title_text=measure, row=1, col=measure_ct + 1,
            )
            event_comparisons_fig.update_xaxes(
                categoryorder="array",
                categoryarray=list(dataset_order.keys()),
                row=1,
                col=measure_ct + 1,
            )

    event_comparisons_fig.update_layout(
        boxmode="group", title_text="MC vs. TC Event Kinetics", title_x=0.5,
    )

    # event_comparisons_fig.show()

    return event_comparisons_fig


def plot_annotated_trace(trace, annotation_values, genotype):
    """
    Takes the trace from a single sweep and plots it on a smaller timescale
    to demonstrate response onset latency, time to peak, and peak amplitude.
    """
    onset_time = annotation_values[0]
    onset_latency = annotation_values[1]
    peak_amp = annotation_values[2]
    time_topeak = annotation_values[3]

    onset_amp = trace[onset_time]

    peak_time = onset_time + time_topeak

    layout = go.Layout(plot_bgcolor="rgba(0,0,0,0)")
    trace_to_plot = trace[518:530]

    color = {"OMP": "#ff9300", "Gg8": "#7a81ff"}

    annotated_plot = go.Figure(layout=layout)

    annotated_plot.add_trace(
        go.Scatter(
            x=trace_to_plot.index,
            y=trace_to_plot,
            # name=type,
            mode="lines",
            line=dict(color=color[genotype], width=4),
            # legendgroup=duration,
        )
    )

    # adds line for light stim
    annotated_plot.add_shape(
        type="rect",
        x0=520,
        y0=35,
        x1=521,
        y1=40,
        line=dict(color="#33F7FF"),
        fillcolor="#33F7FF",
    )

    # adds annotation for onset latency
    annotated_plot.add_annotation(
        x=onset_time + 1.5,
        y=onset_amp,
        text="Response onset:<br>{} ms latency".format(
            round(onset_latency, 1)
        ),
        font=dict(size=24),
        # align="left",
        showarrow=False,
        xshift=50,
    )

    annotated_plot.add_trace(
        go.Scatter(
            x=[onset_time],
            y=[onset_amp],
            mode="markers",
            marker=dict(size=20, color="#CF50C6")
            # text="Response onset",
            # textposition="middle right",
            # textfont=dict(size=20),
        )
    )

    # annotated_plot.update_layout(autosize=False, margin=dict(b=100))

    # adds annotation for peak amplitude
    annotated_plot.add_annotation(
        x=peak_time + 1,
        # y=peak_amp,
        yref="paper",
        y=-0.15,
        text="Peak amplitude:<br>{} pA".format(round(peak_amp)),
        font=dict(size=24),
        # align="left",
        showarrow=False,
        # yshift=-100,
    )

    annotated_plot.add_trace(
        go.Scatter(
            x=[peak_time],
            y=[peak_amp],
            mode="markers",
            marker=dict(size=20)
            # text="Response onset",
            # textposition="middle right",
            # textfont=dict(size=20),
        )
    )

    # add line and annotation for time to peak
    annotated_plot.add_shape(
        type="line",
        x0=onset_time,
        y0=peak_amp,
        x1=peak_time,
        y1=peak_amp,
        line=dict(dash="dash", width=3, color="#33B1FF"),
    )
    annotated_plot.add_annotation(
        # x=(peak_time - onset_time - 2) / 2 + (onset_time - 2),
        x=onset_time,
        y=peak_amp,
        text="Time to peak:<br>{} ms".format(round(time_topeak, 1)),
        showarrow=False,
        # yshift=50,
        xshift=-70,
        font=dict(size=24, family="Arial"),
    )

    # adds horizontal line + text for scale bar
    annotated_plot.add_shape(
        type="line", x0=527, y0=-300, x1=529, y1=-300,
    )
    annotated_plot.add_annotation(
        x=528,
        y=-300,
        yshift=-18,
        text="2 ms",
        showarrow=False,
        font=dict(size=20),
    )

    # adds vertical line + text for scale bar
    annotated_plot.add_shape(type="line", x0=529, y0=-300, x1=529, y1=-200)

    annotated_plot.add_annotation(
        x=529,
        y=-250,
        xshift=40,
        text="100 pA",
        showarrow=False,
        # textangle=-90,
        font=dict(size=20),
    )

    annotated_plot.update_layout(font=dict(family="Arial",), showlegend=False)

    annotated_plot_noaxes = go.Figure(annotated_plot)
    annotated_plot_noaxes.update_xaxes(showgrid=False, visible=False)
    annotated_plot_noaxes.update_yaxes(showgrid=False, visible=False)

    # annotated_plot.show()
    # annotated_plot_noaxes.show()

    return annotated_plot, annotated_plot_noaxes


def save_annotated_figs(axes, noaxes, cell, genotype):
    """
    Saves the example traces figs as static png file
    """

    if not os.path.exists(FileSettings.PAPER_FIGURES_FOLDER):
        os.makedirs(FileSettings.PAPER_FIGURES_FOLDER)

    axes_filename = "{}_{}_trace_annotated.png".format(
        cell.cell_name, genotype
    )
    noaxes_filename = "{}_{}_trace_annotated_noaxes.png".format(
        cell.cell_name, genotype
    )

    axes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, axes_filename)
    )

    noaxes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, noaxes_filename)
    )


def make_one_plot_trace(file_name, cell_trace, type, inset=False):
    """
    Makes the trace data used to plot later. "type" parameter determines the 
    color of the trace
    
    """
    intensity, duration = FileSettings.SELECTED_CONDITION

    if file_name == "JH20210923_c2.nwb":
        intensity = "50%"
        duration = " 1 ms"

    if type == "NBQX":
        trace_to_plot = cell_trace.loc[:, 500.00:700.00]
        trace_y_toplot = cell_trace.loc[500.00:700.00].squeeze()

    else:
        # sets cell trace values
        stim_columns = cell_trace.loc[:, ["Light Intensity", "Light Duration"]]
        trace_to_plot = cell_trace.loc[
            :, 500.00:700.00
        ]  # only plots first 400-1000 ms

        trace_to_plot_combined = pd.concat(
            [stim_columns, trace_to_plot], axis=1
        )

        trace_y_toplot = trace_to_plot_combined.loc[
            (trace_to_plot_combined["Light Intensity"] == intensity)
            & (trace_to_plot_combined["Light Duration"] == duration),
            500.00::,
        ].squeeze()

    color = {
        "Control": "#414145",
        "NBQX": "#EE251F",
        "OMP cell 1": "#ff9300",
        "OMP cell 2": "#FBB85C",
        "Gg8 cell 1": "#7a81ff",
        "Gg8 cell 2": "#A4A8F9",
    }
    if inset is True:
        plot_trace = go.Scatter(
            x=trace_y_toplot.index
            if type == "NBQX"
            else trace_to_plot.columns,
            y=trace_y_toplot,
            xaxis="x2",
            yaxis="y2",
            name=type,
            mode="lines",
            line=dict(color=color[type], width=2),
            # legendgroup=duration,
        )
    else:
        plot_trace = go.Scatter(
            x=trace_to_plot.columns,
            y=trace_y_toplot,
            name=type,
            mode="lines",
            line=dict(color=color[type], width=4),
            # legendgroup=duration,
        )

    return plot_trace


def make_inset_plot_fig(
    genotype, main_trace1, main_trace2, inset_trace1, inset_trace2
):
    """
    Takes four traces and makes a main plot with inset plot
    """
    data = [main_trace1, main_trace2, inset_trace1, inset_trace2]

    # sets background color to white
    layout = go.Layout(
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-700, 150],),
        xaxis2=dict(domain=[0.55, 0.95], anchor="y2"),
        yaxis2=dict(domain=[0.1, 0.5], anchor="x2"),
    )

    inset_plot = go.Figure(data=data, layout=layout)

    # adds line for light stim
    inset_plot.add_shape(
        type="rect",
        x0=520,
        y0=50,
        x1=521,
        y1=100,
        line=dict(color="#33F7FF"),
        fillcolor="#33F7FF",
    )

    # adds horizontal line + text for main plot scale bar
    inset_plot.add_shape(
        type="line", x0=530, y0=-600, x1=555, y1=-600,
    )
    inset_plot.add_annotation(
        x=542.5, y=-650, text="25 ms", showarrow=False, font=dict(size=20)
    )

    # adds vertical line + text for main plot scale bar
    inset_plot.add_shape(type="line", x0=555, y0=-600, x1=555, y1=-400)

    inset_plot.add_annotation(
        x=575,
        y=-500,
        text="200 pA",
        showarrow=False,
        # textangle=-90,
        font=dict(size=20),
    )

    # adds horizontal line + text for inset plot scale bar
    inset_plot.add_shape(
        xref="x2",
        yref="y2",
        type="line",
        x0=600,
        y0=-300 if genotype == "OMP" else -35,
        x1=620,
        y1=-300 if genotype == "OMP" else -35,
    )
    inset_plot.add_annotation(
        xref="x2",
        yref="y2",
        x=610,
        y=-380 if genotype == "OMP" else -40,
        text="20 ms",
        showarrow=False,
        font=dict(size=16),
    )

    # adds vertical line + text for inset plot scale bar
    inset_plot.add_shape(
        xref="x2",
        yref="y2",
        type="line",
        x0=620,
        y0=-300 if genotype == "OMP" else -35,
        x1=620,
        y1=-200 if genotype == "OMP" else -25,
    )

    inset_plot.add_annotation(
        xref="x2",
        yref="y2",
        x=650 if genotype == "Gg8" else 660,
        y=-250 if genotype == "OMP" else -30,
        text="100 pA" if genotype == "OMP" else "10 pA",
        showarrow=False,
        # textangle=-90,
        font=dict(size=16),
    )

    # add box around inset plot
    inset_plot.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.53,
        y0=0.1,
        x1=0.97,
        y1=0.5,
        line={"width": 1, "color": "black"},
    )

    # inset_plot.update_shapes(dict(xref="x", yref="y"))

    inset_plot.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        gridcolor="black",
        ticks="outside",
        tick0=520,
        dtick=10,
    )
    inset_plot.update_yaxes(
        showline=True, linewidth=1, gridcolor="black", linecolor="black",
    )

    inset_plot.update_layout(
        font_family="Arial", legend=dict(font=dict(family="Arial", size=26))
    )

    inset_plot_noaxes = go.Figure(inset_plot)
    inset_plot_noaxes.update_xaxes(showgrid=False, visible=False)
    inset_plot_noaxes.update_yaxes(showgrid=False, visible=False)

    # inset_plot.show()
    # inset_plot_noaxes.show()

    # pdb.set_trace()

    return inset_plot, inset_plot_noaxes


def save_example_traces_figs(axes, noaxes, genotype):
    """
    Saves the example traces figs as static png file
    """

    if not os.path.exists(FileSettings.PAPER_FIGURES_FOLDER):
        os.makedirs(FileSettings.PAPER_FIGURES_FOLDER)

    axes_filename = "{}_example_traces.png".format(genotype)
    noaxes_filename = "{}_example_traces_noaxes.png".format(genotype)

    axes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, axes_filename)
    )

    noaxes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, noaxes_filename)
    )


def plot_spike_sweeps(genotype, trace):
    """
    Plots a single spike sweep to show STC physiology
    """
    color = {"OMP": "#ff9300", "Gg8": "#7a81ff"}

    layout = Layout(plot_bgcolor="rgba(0,0,0,0)")
    to_plot = trace[400:1600]
    zoomed_to_plot = trace[530:605]

    spikes_plots = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.05
    )
    # spikes_plots = go.Figure(layout=layout)
    # pdb.set_trace()

    # add main spike train
    spikes_plots.add_trace(
        go.Scatter(
            x=to_plot.index,
            y=to_plot,
            # name=type_names[0],
            mode="lines",
            line=dict(
                # color=color[genotype],
                color="#414145",
                width=2,
            ),
            # legendgroup=duration,
        ),
        row=1,
        col=1,
    )

    # add zoomed-in spikes
    spikes_plots.add_trace(
        go.Scatter(
            x=zoomed_to_plot.index,
            y=zoomed_to_plot,
            # name=type_names[0],
            mode="lines",
            line=dict(
                # color=color[genotype],
                color="#414145",
                width=2,
            ),
            # legendgroup=duration,
        ),
        row=1,
        col=2,
    )

    spikes_plots.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        gridcolor="black",
        ticks="outside",
        tick0=400,
        dtick=100,
    )

    spikes_plots.update_yaxes(
        showline=True, linewidth=1, gridcolor="black", linecolor="black",
    )

    # add shaded box around spikes that we're zooming in on
    spikes_plots.add_shape(
        type="rect",
        xref="x1",
        yref="y1",
        x0=528,
        y0=-50,
        x1=607,
        y1=25,
        line=dict(color="#B1EE81"),
        fillcolor="#B1EE81",
        opacity=0.5,
        layer="below",
    )

    # add shaded border around zoomed in subplot
    spikes_plots.add_shape(
        type="rect",
        xref="x2",
        yref="y2",
        x0=525,
        y0=-52,
        x1=610,
        y1=25,
        line=dict(color="#B1EE81"),
    )

    # adds horizontal line + text for main spikes scale bar
    spikes_plots.add_shape(
        type="line", xref="x1", yref="y1", x0=1400, y0=-10, x1=1600, y1=-10,
    )
    spikes_plots.add_annotation(
        xref="x1",
        yref="y1",
        x=1500,
        y=-10,
        yshift=-20,
        text="200 ms",
        showarrow=False,
        font=dict(size=20),
    )

    # adds vertical line + text for main spikes scale bar
    spikes_plots.add_shape(
        type="line", xref="x1", yref="y1", x0=1600, y0=-10, x1=1600, y1=10
    )

    spikes_plots.add_annotation(
        xref="x1",
        yref="y1",
        x=1600,
        y=0,
        xshift=40,
        text="20 mV",
        showarrow=False,
        # textangle=-90,
        font=dict(size=20),
    )

    # add arrow annotation for Vr
    spikes_plots.add_annotation(
        xref="x1",
        yref="y1",
        x=450,
        y=to_plot[450],
        # xshift=25,
        yshift=5,
        # text="{} mV".format(round(to_plot[450])),
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
    )

    # add text annotation for Vr
    spikes_plots.add_annotation(
        xref="x1",
        yref="y1",
        x=450,
        y=to_plot[450],
        yshift=40,
        xshift=-10,
        text="{} mV".format(round(to_plot[450])),
        showarrow=False,
        font=dict(size=20),
    )

    # adds horizontal line + text for zoomed spikes scale bar
    spikes_plots.add_shape(
        type="line", xref="x2", yref="y2", x0=612.5, y0=0, x1=637.5, y1=0,
    )
    spikes_plots.add_annotation(
        xref="x2",
        yref="y2",
        x=625,
        y=0,
        yshift=-20,
        text="25 ms",
        showarrow=False,
        font=dict(size=20),
    )

    # adds vertical line + text for zoomed spikes scale bar
    spikes_plots.add_shape(
        type="line", xref="x2", yref="y2", x0=637.5, y0=0, x1=637.5, y1=10
    )

    spikes_plots.add_annotation(
        xref="x2",
        yref="y2",
        x=637.5,
        y=5,
        xshift=40,
        text="10 mV",
        showarrow=False,
        # textangle=-90,
        font=dict(size=20),
    )

    spikes_plots.update_layout(
        font_family="Arial",
        showlegend=False,
        width=1200,
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    spikes_plots_noaxes = go.Figure(spikes_plots)
    spikes_plots_noaxes.update_xaxes(showgrid=False, visible=False)
    spikes_plots_noaxes.update_yaxes(showgrid=False, visible=False)

    # spikes_plots_noaxes.show()
    # pdb.set_trace()

    return spikes_plots, spikes_plots_noaxes


def save_spike_figs(axes, noaxes, cell, genotype):
    """
    Saves the example traces figs as static png file
    """

    if not os.path.exists(FileSettings.PAPER_FIGURES_FOLDER):
        os.makedirs(FileSettings.PAPER_FIGURES_FOLDER)

    axes_filename = "{}_{}_spikes.png".format(cell.cell_name, genotype)
    noaxes_filename = "{}_{}_spikes_noaxes.png".format(
        cell.cell_name, genotype
    )

    axes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, axes_filename)
    )

    noaxes.write_image(
        os.path.join(FileSettings.PAPER_FIGURES_FOLDER, noaxes_filename)
    )


def plot_power_curve_traces(mean_trace_df, sweeps_df):
    """
    Plots the baseline-subtracted mean trace for each stimulus condition around the response time, 
    one subplot for each duration, if applicable. Does this for one cell.
    """

    # intensities and durations, and color might need to become self variables

    sweep_analysis_values = sweeps_df
    intensities = sweep_analysis_values["Light Intensity"].unique()
    durations = sweep_analysis_values["Light Duration"].unique()

    # blue colors
    color = ["#0859C6", "#10A5F5", "#00DBFF"]

    stim_columns = mean_trace_df.loc[:, ["Light Intensity", "Light Duration"]]
    traces_to_plot = mean_trace_df.loc[
        :, 500.00:700.00
    ]  # only plots first 400-1000 ms
    traces_to_plot_combined = pd.concat([stim_columns, traces_to_plot], axis=1)

    power_curve_traces = make_subplots(
        # rows=len(intensities), cols=1,
        rows=1,
        cols=len(intensities),
        subplot_titles=(intensities[::-1]),
        shared_yaxes=True,
        x_title="Time (ms)",
        y_title="Amplitude (pA)",
    )

    # new method for hiding duplicate legends:
    # create a list to track each time a duration has been plotted, and only show legends
    # for the first time the duration is plotted
    duration_checker = []

    for intensity_count, intensity in enumerate(intensities):
        for duration_count, duration in enumerate(durations):

            # plot sweeps from all intensities of one duration
            y_toplot = traces_to_plot_combined.loc[
                (traces_to_plot_combined["Light Intensity"] == intensity)
                & (traces_to_plot_combined["Light Duration"] == duration),
                500.00::,
            ].squeeze()
            power_curve_traces.add_trace(
                go.Scatter(
                    x=traces_to_plot.columns,
                    y=y_toplot,
                    name=duration,
                    mode="lines",
                    line=dict(color=color[duration_count]),
                    showlegend=False if duration in duration_checker else True,
                    legendgroup=duration,
                ),
                # row=intensity_count+1, col=1
                row=1,
                col=len(intensities) - intensity_count,
            )
            if len(y_toplot) != 0:
                duration_checker.append(duration)

    # below is code from stack overflow to hide duplicate legends
    # names = set()
    # mean_traces_fig.for_each_trace(
    #     lambda trace:
    #         trace.update(showlegend=False)
    #         if (trace.name in names) else names.add(trace.name))

    power_curve_traces.update_layout(
        title_text="Light Intensity",
        title_x=0.45,
        legend_title_text="Light Duration",
        title_font=dict(size=20, family="Arial"),
        legend=dict(font=dict(family="Arial", size=16)),
    )

    power_curve_traces.update_xaxes(
        title_font=dict(size=24, family="Arial"),
        tickfont=dict(size=16, family="Arial"),
        tickangle=45,
        automargin=True,
        autorange=True,
    )

    power_curve_traces.update_yaxes(
        title_font=dict(size=24, family="Arial"),
        tickfont=dict(size=16, family="Arial"),
        tick0=500,
        dtick=100,
        automargin=True,
    )

    power_curve_traces.update_annotations(font_size=20, font_family="Arial")
    # power_curve_traces.show()

    return power_curve_traces


def save_power_curve_traces(genotype, cell_name, fig):
    """
    Saves the power curve traces figs as static png file
    """

    if not os.path.exists(FileSettings.PAPER_FIGURES_FOLDER):
        os.makedirs(FileSettings.PAPER_FIGURES_FOLDER)

    filename = "{}_{}_power_curve_traces.png".format(cell_name, genotype)

    fig.write_image(os.path.join(FileSettings.PAPER_FIGURES_FOLDER, filename))


def graph_power_curve(power_curve_stats, sweep_analysis_values):
    """
        do a loop through available durations and intensities instead of hard
        coding. maybe need MultiIndex after all?? Put power curve + all stats 
        measurements in subplots
        """

    intensities = sweep_analysis_values["Light Intensity"].unique()
    durations = sweep_analysis_values["Light Duration"].unique()

    color = ["#0859C6", "#10A5F5", "#00DBFF"]

    power_curve = go.Figure()

    # make the x-axis of light intensity to be used in each subplot

    x_sweep_dict = {}

    for duration in durations:
        x_sweep_intensity = sweep_analysis_values.loc[
            sweep_analysis_values["Light Duration"] == duration,
            ["Light Intensity"],
        ]

        x_sweep_dict[duration] = x_sweep_intensity

    # pdb.set_trace()
    for count, duration in enumerate(durations):

        error = power_curve_stats.loc[
            power_curve_stats["Light Duration"] == duration, ["SEM"]
        ].squeeze()

        if len(intensities) > 1:
            if isinstance(error, float) != True:
                # only make power curve if more than one intensity exists

                # power curve
                power_curve.add_trace(
                    go.Scatter(
                        x=power_curve_stats.loc[
                            power_curve_stats["Light Duration"] == duration,
                            ["Light Intensity"],
                        ].squeeze(),
                        y=power_curve_stats.loc[
                            power_curve_stats["Light Duration"] == duration,
                            ["Mean Response Amplitude (pA)"],
                        ].squeeze(),
                        name=duration,
                        error_y=dict(
                            type="data", array=error.values, visible=True
                        ),
                        mode="lines+markers",
                        line=dict(color=color[count]),
                        legendgroup=duration,
                    ),
                )

        # Update xaxis properties
        # curve_stats_fig.update_xaxes(autorange="reversed")
        # this defines the intensities order for x-axes
        power_curve.update_xaxes(
            title_text="Light Intensity",
            categoryorder="array",
            categoryarray=np.flip(intensities),
            title_font=dict(size=20, family="Arial"),
            tickfont=dict(size=16, family="Arial"),
        )

        # Update yaxis properties
        power_curve.update_yaxes(
            title_text="Response Amplitude (pA)",
            autorange="reversed",
            title_font=dict(size=20, family="Arial"),
            tickfont=dict(size=16, family="Arial"),
        )

    power_curve.update_layout(
        # yaxis_title='Onset Latency (ms)',
        boxmode="group"  # group together boxes of the different traces for each value of x
    )

    # below is code from stack overflow to hide duplicate legends
    names = set()
    power_curve.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    power_curve.update_layout(
        legend_title_text="Light Duration",
        font=dict(family="Arial", size=20),
        legend=dict(font=dict(family="Arial", size=16)),
    )
    power_curve.update_annotations(font_size=20, font_family="Arial")

    # power_curve.show()

    return power_curve


def save_power_curve(genotype, cell_name, fig):
    """
    Saves the power curve traces figs as static png file
    """

    if not os.path.exists(FileSettings.PAPER_FIGURES_FOLDER):
        os.makedirs(FileSettings.PAPER_FIGURES_FOLDER)

    filename = "{}_{}_power_curve.png".format(cell_name, genotype)

    fig.write_image(os.path.join(FileSettings.PAPER_FIGURES_FOLDER, filename))


def make_event_hist_trace(trace_color, condition, data):
    """
    Makes the plotting traces for event kinetics histogram, for events within 
    response window and outside of response window
    """

    # plots amplitude as histogram

    hist_trace = go.Histogram(
        x=data,
        marker_color=trace_color,
        name=condition,
        legendgroup=condition,
        bingroup=condition,
    )

    return hist_trace


def add_median_vline(hist, annotation_color, data, win_count, count, unit):
    """
    Adds the vertical line marking the median of histograms, then returns the
    histogram
    """
    median = data.median()

    hist.add_vline(
        row=1,
        col=win_count + 1,
        x=median,
        line_color=annotation_color,
        annotation_text=f"median = {round(median, 2)} {unit}",
        annotation_font=dict(color=annotation_color),
        annotation_yshift=-10 if count == 1 else 0,
        annotation_position="top right",
    )


def plot_response_win_comparison(
    dataset, cell_type, cell_name, stim_time, colors, stats_dict
):
    """
    Plots the response windows used for stats comparison to determine whether cell
    has a response or not.
    """

    nested_dict = lambda: defaultdict(
        nested_dict
    )  # put titles in dict instead
    titles_dict = nested_dict()
    for comparison in list(stats_dict):
        for subtraction_type in list(stats_dict[comparison]):
            ttest_pval = stats_dict[comparison][subtraction_type]["ttest pval"]
            ks_pval = stats_dict[comparison][subtraction_type]["ks pval"]

            # rounds p-values for easier readaibility
            if ttest_pval < 0.01:
                ttest_pval = f"{ttest_pval:.2e}"
            else:
                ttest_pval = np.round(ttest_pval, 4)

            if ks_pval < 0.01:
                ks_pval = f"{ks_pval:.2e}"
            else:
                ks_pval = np.round(ks_pval, 4)

            title = (
                f"{comparison}, {subtraction_type} <br>"
                f"t-test pval = {ttest_pval}, ks pval = {ks_pval}"
            )
            titles_dict[comparison][subtraction_type]["title"] = title

    stats_fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "temp subtitle"
            for subplot in range(
                len(list(stats_dict)) * len(list(stats_dict[comparison]))
            )
        ],
        x_title="Time (ms)",
        y_title="Avg Frequency (Hz)",
        shared_xaxes=True,
        shared_yaxes=True,
    )

    stats_fig.update_layout(
        title_text=(f"{dataset} {cell_type} {cell_name} Response Comparisons"),
        title_x=0.5,
    )

    subplot_index = 0
    for comp_count, comparison in enumerate(list(stats_dict)):
        for sub_count, subtraction_type in enumerate(
            list(stats_dict[comparison])
        ):
            x_name = comparison.split(" vs. ")[0]
            y_name = comparison.split(" vs. ")[1]

            # plots x freq
            stats_fig.add_trace(
                go.Scatter(
                    x=stats_dict[comparison][subtraction_type]["freqs"][
                        "x"
                    ].index,
                    y=stats_dict[comparison][subtraction_type]["freqs"]["x"],
                    name=x_name,
                    marker=dict(color=colors[x_name]),
                    legendgroup=x_name,
                ),
                row=comp_count + 1,
                col=sub_count + 1,
            )

            # plots y freq
            stats_fig.add_trace(
                go.Scatter(
                    x=stats_dict[comparison][subtraction_type]["freqs"][
                        "y"
                    ].index,
                    y=stats_dict[comparison][subtraction_type]["freqs"]["y"],
                    name=y_name,
                    marker=dict(color=colors[y_name]),
                    legendgroup=y_name,
                ),
                row=comp_count + 1,
                col=sub_count + 1,
            )

            # updates subplot title using titles_dict
            stats_fig.layout.annotations[subplot_index]["text"] = titles_dict[
                comparison
            ][subtraction_type]["title"]
            subplot_index += 1
            # stats_fig.update_layout(
            #     title=titles_dict[comparison][subtraction_type]["title"],
            #     row=sub_count + 1,
            #     col=comp_count + 1,
            # )

    # adds line for light stim
    stats_fig.add_vrect(
        type="rect",
        x0=stim_time,
        x1=stim_time + 100,
        fillcolor="#33F7FF",
        opacity=0.5,
        layer="below",
        line_width=0,
    )

    # below is code from stack overflow to hide duplicate legends
    names = set()
    stats_fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    # stats_fig.show()

    return stats_fig

