import os
import pandas as pd
from collections import defaultdict

from regex import F
from run_single_test import run_both_conditions
from aggregate_stats import *
from file_settings import FileSettings

from single_test import JaneCell

from plotting import *

# from ppt_plotting import *
import pdb


def get_datasets():
    # dataset_list = ["p2"]
    dataset_list = [
        dataset
        for dataset in os.listdir(FileSettings.DATA_FOLDER)
        if dataset not in FileSettings.IGNORED
    ]
    return dataset_list


def get_data_info():
    # gets the list of datasets from file directory
    dataset_list = get_datasets()
    # dataset_list = ["5dpi"]

    # runs stats analysis for each dataset

    recorded_counts = get_patched_counts(dataset_list)
    all_patched = make_patched_counts_df(recorded_counts)

    return dataset_list, all_patched


def run_dataset_analysis(dataset):
    dataset_data_folder = os.path.join(FileSettings.DATA_FOLDER, dataset)

    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(FileSettings.TABLES_FOLDER, dataset, csvfile_name)

    ibwfile_list = []

    # only adds light file
    for file in os.listdir(dataset_data_folder):
        if file.endswith("light100.ibw"):
            ibwfile_list.append(file)

    for file_count, ibwfile_name in enumerate(ibwfile_list):
        cell_name = ibwfile_name.split("_light")[0]
        # run_both_conditions(dataset, csvfile, cell_name)
        print(
            f"Analysis for {cell_name} done, "
            f"{file_count+1}/{len(ibwfile_list)} cells"
        )


def collect_dataset_stats(dataset_list):
    # this replicates previous old analysis
    dataset_all_mean_trace_stats = defaultdict(dict)

    dataset_cell_counts = defaultdict(dict)
    dataset_mean_trace_stats = defaultdict(dict)
    dataset_median_stats = defaultdict(dict)
    dataset_freq_stats = defaultdict(dict)

    for dataset_count, dataset in enumerate(dataset_list):
        print("***Starting analysis for {} dataset.***".format(dataset))
        run_dataset_analysis(dataset)
        cell_types_list = get_cell_types(dataset)
        (
            all_mean_trace_stats,
            response_cells_list,
            mean_trace_stats,
            response_counts,
            median_stats,
            freq_stats,
        ) = get_cell_type_summary(dataset, cell_types_list)

        dataset_all_mean_trace_stats[dataset] = all_mean_trace_stats

        dataset_cell_counts[dataset] = response_counts
        dataset_mean_trace_stats[dataset] = mean_trace_stats
        dataset_median_stats[dataset] = median_stats
        dataset_freq_stats[dataset] = freq_stats

        (
            all_mean_trace_stats_df,
            dataset_counts_df,
            mean_trace_stats_df,
            win_median_stats_df,
            outside_median_stats_df,
            freq_stats_df,
        ) = make_cell_type_summary_dfs(
            dataset,
            all_mean_trace_stats,
            cell_types_list,
            mean_trace_stats,
            response_counts,
            median_stats,
            freq_stats,
        )

        save_dataset_stats(
            dataset,
            all_mean_trace_stats_df,
            response_cells_list,
            mean_trace_stats_df,
            dataset_counts_df,
            win_median_stats_df,
            outside_median_stats_df,
            freq_stats_df,
        )

        print(
            "***Analysis for {} dataset done, #{}/{} datasets.***".format(
                dataset, dataset_count + 1, len(dataset_list)
            )
        )

    return (
        cell_types_list,
        dataset_all_mean_trace_stats,
        dataset_cell_counts,
        dataset_mean_trace_stats,
        dataset_median_stats,
        dataset_freq_stats,
    )


def make_summary_plots(
    cell_types_list,
    dataset_all_mean_trace_stats,
    dataset_cell_counts,
    dataset_mean_trace_stats,
    dataset_median_stats,
    dataset_freq_stats,
):
    # now we plot using dicts
    response_fig, response_fig_data = plot_response_counts(dataset_cell_counts)

    (
        windowed_event_medians_figs,
        medians_fig_data,
    ) = plot_windowed_median_event_stats(dataset_median_stats, cell_types_list)
    (
        event_comparisons_fig,
        event_comparisons_fig_data,
    ) = plot_cell_type_event_comparisons(dataset_median_stats)

    # plots mean trace stats
    mean_trace_stats_fig, mean_trace_stats_fig_data = plot_mean_trace_stats(
        dataset_mean_trace_stats
    )

    # plots all mean trace stats
    (
        all_mean_trace_stats_fig,
        all_mean_trace_stats_fig_data,
    ) = plot_mean_trace_stats(dataset_all_mean_trace_stats)
    # plots avg freq stats
    freq_stats_fig, freq_stats_fig_data = plot_freq_stats(dataset_freq_stats)

    return (
        all_mean_trace_stats_fig,
        all_mean_trace_stats_fig_data,
        response_fig,
        response_fig_data,
        windowed_event_medians_figs,
        medians_fig_data,
        event_comparisons_fig,
        event_comparisons_fig_data,
        mean_trace_stats_fig,
        mean_trace_stats_fig_data,
        freq_stats_fig,
        freq_stats_fig_data,
    )


def save_summary_plots(
    all_mean_trace_stats_fig,
    all_mean_trace_stats_fig_data,
    response_fig,
    response_fig_data,
    windowed_event_medians_figs,
    medians_fig_data,
    event_comparisons_fig,
    event_comparisons_fig_data,
    mean_trace_stats_fig,
    mean_trace_stats_fig_data,
    freq_stats_fig,
    freq_stats_fig_data,
):

    # save as html, also saves plotted data as csvs
    save_all_mean_trace_fig(
        all_mean_trace_stats_fig, all_mean_trace_stats_fig_data
    )

    save_response_counts_fig(response_fig, response_fig_data)
    save_median_events_fig(
        windowed_event_medians_figs,
        event_comparisons_fig,
        medians_fig_data,
        event_comparisons_fig_data,
    )
    save_freq_mean_trace_figs(
        mean_trace_stats_fig,
        freq_stats_fig,
        mean_trace_stats_fig_data,
        freq_stats_fig_data,
    )

    # change rows to 1.1 if using ppt_plotting
    # save to png
    save_fig_to_png(
        all_mean_trace_stats_fig,
        legend=True,
        rows=1,
        cols=3,
        png_filename="all_mean_trace_stats.png",
    )

    save_fig_to_png(
        response_fig,
        legend=True,
        rows=1,
        cols=2,
        png_filename="all_response_counts.png",
    )

    # change rows to 1.1 if using ppt_plotting
    save_fig_to_png(
        mean_trace_stats_fig,
        legend=True,
        rows=1,
        cols=3,
        png_filename="mean_trace_stats.png",
    )
    save_fig_to_png(
        freq_stats_fig,
        legend=True,
        rows=1,
        cols=3,
        png_filename="avg_freq_stats.png",
    )

    save_fig_to_png(
        windowed_event_medians_figs[0],
        legend=True,
        rows=3,
        cols=2,
        png_filename="p2_windowed_event_medians.png",
    )

    save_fig_to_png(
        windowed_event_medians_figs[1],
        legend=True,
        rows=3,
        cols=2,
        png_filename="p14_windowed_event_medians.png",
    )

    save_fig_to_png(
        event_comparisons_fig,
        legend=True,
        rows=1,
        cols=3,
        png_filename="cell_type_event_comparisons.png",
    )


def main():
    dataset_list, all_patched = get_data_info()
    dataset_list.sort(reverse=True)  # put p2 first

    (
        cell_types_list,
        dataset_all_mean_trace_stats,
        dataset_cell_counts,
        dataset_mean_trace_stats,
        dataset_median_stats,
        dataset_freq_stats,
    ) = collect_dataset_stats(dataset_list)

    (
        all_mean_trace_stats_fig,
        all_mean_trace_stats_fig_data,
        response_fig,
        response_fig_data,
        windowed_event_medians_fig,
        medians_fig_data,
        event_comparisons_fig,
        event_comparisons_fig_data,
        mean_trace_stats_fig,
        mean_trace_stats_fig_data,
        freq_stats_fig,
        freq_stats_fig_data,
    ) = make_summary_plots(
        cell_types_list,
        dataset_all_mean_trace_stats,
        dataset_cell_counts,
        dataset_mean_trace_stats,
        dataset_median_stats,
        dataset_freq_stats,
    )

    (
        all_mean_trace_avgsem,
        windowed_median_avgsem,
        event_comparisons_avgsem,
        mean_trace_avgsem,
        freq_avgsem,
    ) = get_avg_sem(
        all_mean_trace_stats_fig_data,
        medians_fig_data,
        event_comparisons_fig_data,
        mean_trace_stats_fig_data,
        freq_stats_fig_data,
    )

    save_avg_sem(
        all_mean_trace_avgsem,
        windowed_median_avgsem,
        event_comparisons_avgsem,
        mean_trace_avgsem,
        freq_avgsem,
    )
    save_summary_plots(
        all_mean_trace_stats_fig,
        all_mean_trace_stats_fig_data,
        response_fig,
        response_fig_data,
        windowed_event_medians_fig,
        medians_fig_data,
        event_comparisons_fig,
        event_comparisons_fig_data,
        mean_trace_stats_fig,
        mean_trace_stats_fig_data,
        freq_stats_fig,
        freq_stats_fig_data,
    )


if __name__ == "__main__":

    main()
