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

import plotly.io as pio

pio.renderers.default = "browser"

# from acq_parameters import *
import pdb
from single_test import JaneCell


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
    # cell.plot_events()
    # cell.plot_event_raster()
    cell.plot_mean_trace()

    # cell.make_cell_analysis_dict()

    # 5 calculates power curve for plotting
    cell.make_power_curve_stats_df()

    # 6 calculates response stats for plotting
    cell.make_stats_df()

    # 7 plots mean traces
    cell.make_mean_traces_df()
    cell.graph_response_trace()

    # 8 makes plots for power curve and response stats if cell responds
    if response == True:

        summary_plots = cell.graph_curve_stats()
        cell.export_stats_csv()
    else:
        print("Cell doesn't have response, no response stats plotted")

    # 9 saves combined plots as html file, exports stats as csv
    cell.output_html_plots()


if __name__ == "__main__":
    dataset = "p2"
    csvfile_name = "{}_data_notes.csv".format(dataset)
    csvfile = os.path.join(
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables",
        dataset,
        csvfile_name,
    )
    file_name = "JH200313_c3_light100.ibw"

    run_single(dataset, csvfile, file_name)

    print("Analysis for {} done".format(file_name))
