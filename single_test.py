from pickle import FALSE
import pandas as pd
import matplotlib.pyplot as plt
import pynwb
import os
import numpy as np
from neo.io import IgorIO
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from pynwb import NWBHDF5IO
import elephant
import scipy
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import sem
from scipy import io

import pymc3 as pm
from theano import shared

import plotly.io as pio

pio.renderers.default = "browser"

import p2_acq_parameters
import p14_acq_parameters

# from file_settings import FileSettings
import pdb


def igor_to_pandas(path_to_file):
    """This function opens an igor binary file (.ibw), extracts the time
    series data, and returns a pandas DataFrame"""

    data_raw = IgorIO(filename=path_to_file)
    data_neo = data_raw.read_block()
    data_neo_array = data_neo.segments[0].analogsignals[0]
    data_df = pd.DataFrame(data_neo_array.as_array().squeeze())

    return data_df


class JaneCell(object):
    def __init__(self, dataset, sweep_info, file, file_name):
        self.dataset = dataset
        self.sweep_info = sweep_info
        self.file = file
        self.file_name = file_name
        self.time = None
        self.raw_df = None
        self.raw_ic_df = None
        self.spikes_sweeps_dict = None
        self.drug_sweeps_dict = None
        self.ic_sweeps_dict = None
        self.sweeps_dict = None
        self.mod_events_df = None
        self.event_stats = None
        self.events_fig = None

        self.traces_filtered = None
        self.traces_filtered_sub = None
        self.sub_mean_trace = None
        self.cell_analysis_dict = None
        self.power_curve_df = None
        self.tuples = None
        self.power_curve_stats = None
        self.sweep_analysis_values = None
        self.cell_name = None
        self.cell_type = None
        self.condition = None
        self.response = None

        # set acquisition parameters
        if self.dataset == "p2":
            self.lowpass_freq = p2_acq_parameters.LOWPASS_FREQ
            self.stim_time = p2_acq_parameters.STIM_TIME
            self.post_stim = p2_acq_parameters.POST_STIM
            self.tp_start = p2_acq_parameters.TP_START
            self.tp_length = p2_acq_parameters.TP_LENGTH
            self.vm_jump = p2_acq_parameters.VM_JUMP
            self.pre_tp = p2_acq_parameters.PRE_TP
            self.unit_scaler = p2_acq_parameters.UNIT_SCALER
            self.amp_factor = p2_acq_parameters.AMP_FACTOR
            self.fs = p2_acq_parameters.FS
            self.baseline_start = p2_acq_parameters.BASELINE_START
            self.baseline_end = p2_acq_parameters.BASELINE_END

        elif self.dataset == "p14":
            self.lowpass_freq = p14_acq_parameters.LOWPASS_FREQ
            self.stim_time = p14_acq_parameters.STIM_TIME
            self.post_stim = p14_acq_parameters.POST_STIM
            self.tp_start = p14_acq_parameters.TP_START
            self.tp_length = p14_acq_parameters.TP_LENGTH
            self.vm_jump = p14_acq_parameters.VM_JUMP
            self.pre_tp = p14_acq_parameters.PRE_TP
            self.unit_scaler = p14_acq_parameters.UNIT_SCALER
            self.amp_factor = p14_acq_parameters.AMP_FACTOR
            self.fs = p14_acq_parameters.FS
            self.baseline_start = p14_acq_parameters.BASELINE_START
            self.baseline_end = p14_acq_parameters.BASELINE_END

        self.cell_sweep_info = self.initialize_cell()

    def initialize_cell(self):
        # turn ibw file into a pandas df
        self.raw_df = igor_to_pandas(self.file) * self.amp_factor
        self.traces = self.raw_df

        # gets sweep info for one cell, drops empty values
        file_split = self.file_name.split(".")
        self.cell_name = file_split[0]

        # pdb.set_trace()

        cell_sweep_info = self.sweep_info.loc[
            self.sweep_info["File Path"] == self.file_name
        ]

        self.cell_type = cell_sweep_info["Cell Type"].values[0]

        if "light" in self.file_name:
            self.condition = "light"
        elif "spontaneous" in self.file_name:
            self.condition = "spontaneous"

        return cell_sweep_info

    def check_response(self):
        """
        Checks whether cell is determined to have response or not (from sweep_info)
        """
        if self.cell_sweep_info["Response"].values[0] == "No":
            response = False
        else:
            response = True

        self.response = response

        return response

    # def drop_sweeps(self):
    #     """
    #     If applicable, drops depol sweeps and esc sweeps
    #     """

    #     # pulls the non-escaped sweep numbers out
    #     if "Escaped sweeps" in self.cell_sweep_info.index:
    #         nonesc_range = self.cell_sweep_info.loc["Non-esc sweeps"][0].split(
    #             ","
    #         )
    #         nonesc_sweeps = []
    #         for i in range(len(nonesc_range)):
    #             if "-" in nonesc_range[i]:
    #                 r_start = int(nonesc_range[i].split("-")[0])
    #                 r_end = int(nonesc_range[i].split("-")[1]) + 1
    #                 all_sweeps = list(range(r_start, r_end))
    #                 nonesc_sweeps.extend(all_sweeps)
    #             else:
    #                 nonesc_sweeps.append(int(nonesc_range[i]))

    #     # pdb.set_trace()

    #     # pulls out depol sweep numbers
    #     if "Depol sweeps" in self.cell_sweep_info.index:
    #         if isinstance(self.cell_sweep_info.loc["Depol sweeps"][0], float):
    #             depol_range = str(
    #                 int(self.cell_sweep_info.loc["Depol sweeps"][0])
    #             )
    #         else:
    #             depol_range = self.cell_sweep_info.loc["Depol sweeps"][
    #                 0
    #             ].split(",")
    #         depol_sweeps = []
    #         for i in range(len(depol_range)):
    #             if "-" in depol_range[i]:
    #                 r_start = int(depol_range[i].split("-")[0])
    #                 r_end = int(depol_range[i].split("-")[1]) + 1
    #                 all_sweeps = list(range(r_start, r_end))
    #                 depol_sweeps.extend(all_sweeps)
    #             else:
    #                 depol_sweeps.append(int(depol_range[i]))

    #     # if applicable, drops depol sweeps and  esc sweeps
    #     if "depol_sweeps" in globals():
    #         self.raw_df = self.raw_df.drop(columns=depol_sweeps, axis=1)

    #     if "nonesc_sweeps" in globals():
    #         self.raw_df = self.raw_df.filter(nonesc_sweeps)

    # def check_exceptions(self, stim_sweep_info):
    #     """
    #     Selects specific sweeps/stim conditions for pre-defined cells
    #     """
    #     # JH20211006_c1.nwb only usable traces are 0.01 ms
    #     if self.file_name == "JH20211006_c1.nwb":
    #         stim_sweep_info = stim_sweep_info[
    #             stim_sweep_info.index.str.contains("0.01 ms")
    #         ]
    #         return True, stim_sweep_info

    #     else:
    #         return False, None

    # def make_drug_sweeps_dict(self):
    #     """
    #     Create dict with with stim name as keys, VC data as values - just
    #     for NBQX wash-in sweeps. Only useful for paper_figs script.
    #     """
    #     drug_sweeps_info = self.cell_sweep_info.filter(
    #         like="NBQX wash-in", axis=0
    #     )

    #     drug_sweeps_dict = {}
    #     for i in range(len(drug_sweeps_info.index)):
    #         stim_name = drug_sweeps_info.index[i]
    #         stim_range = drug_sweeps_info.iloc[i].str.split(",")[0]
    #         stim_sweeps = []

    #         for j in range(len(stim_range)):
    #             if "-" in stim_range[j]:
    #                 r_start = int(stim_range[j].split("-")[0])
    #                 r_end = int(stim_range[j].split("-")[1]) + 1
    #                 all_sweeps = list(range(r_start, r_end))
    #                 stim_sweeps.extend(all_sweeps)
    #             else:
    #                 stim_sweeps.append(int(stim_range[j][0]))

    #         stim_sweeps_VC = self.raw_df[
    #             self.raw_df.columns.intersection(set(stim_sweeps))
    #         ]
    #         drug_sweeps_dict[stim_name] = stim_sweeps_VC

    #     # drop keys with empty dataframes
    #     self.drug_sweeps_dict = {
    #         k: v for (k, v) in drug_sweeps_dict.items() if not v.empty
    #     }

    # def make_spikes_dict(self):
    #     """
    #     Collects all the FI IC step sweeps into a dict
    #     """
    #     steps_sweep_info = self.cell_sweep_info.filter(like="FI", axis=0)

    #     sweeps_dict = {}

    #     # define sweep ranges for each stim set present
    #     for i in range(len(steps_sweep_info.index)):
    #         stim_name = steps_sweep_info.index[i]
    #         stim_range = steps_sweep_info.iloc[i].str.split(",")[0]
    #         stim_sweeps = []

    #         for j in range(len(stim_range)):
    #             if "-" in stim_range[j]:
    #                 r_start = int(stim_range[j].split("-")[0])
    #                 r_end = int(stim_range[j].split("-")[1]) + 1
    #                 all_sweeps = list(range(r_start, r_end))
    #                 stim_sweeps.extend(all_sweeps)
    #             else:
    #                 stim_sweeps.append(int(stim_range[j][0]))

    #         stim_sweeps_IC = self.raw_ic_df[
    #             self.raw_ic_df.columns.intersection(set(stim_sweeps))
    #         ]
    #         sweeps_dict[stim_name] = stim_sweeps_IC

    #     # drop keys with empty dataframes
    #     self.ic_sweeps_dict = {
    #         k: v for (k, v) in sweeps_dict.items() if not v.empty
    #     }

    # def make_sweeps_dict(self):
    #     """gv
    #     Create dict with stim name as keys, VC data as values
    #     """

    #     stim_sweep_info = self.cell_sweep_info.filter(like="%", axis=0)
    #     stim_sweep_info = stim_sweep_info[
    #         stim_sweep_info[self.cell_name].str.contains("-")
    #     ]

    #     if self.check_exceptions(stim_sweep_info)[0] == True:
    #         stim_sweep_info = self.check_exceptions(stim_sweep_info)[1]

    #     else:
    #         # drops 0.5 and 0.1 ms sweeps
    #         stim_sweep_info = stim_sweep_info[
    #             ~stim_sweep_info.index.str.contains("0.5 ms")
    #         ]
    #         stim_sweep_info = stim_sweep_info[
    #             ~stim_sweep_info.index.str.contains("0.1 ms")
    #         ]

    #         if self.response == True:
    #             # the dropping doesn't seem to work in one line
    #             stim_sweep_info = stim_sweep_info[
    #                 ~stim_sweep_info.index.str.contains("4 ms")
    #             ]  # drops 4 ms
    #             stim_sweep_info = stim_sweep_info[
    #                 ~stim_sweep_info.index.str.contains("100 ms")
    #             ]  # drops 100 ms
    #     # pdb.set_trace()
    #     sweeps_dict = {}

    #     # define sweep ranges for each stim set present
    #     for i in range(len(stim_sweep_info.index)):
    #         stim_name = stim_sweep_info.index[i]
    #         stim_range = stim_sweep_info.iloc[i].str.split(",")[0]
    #         stim_sweeps = []

    #         for j in range(len(stim_range)):
    #             if "-" in stim_range[j]:
    #                 r_start = int(stim_range[j].split("-")[0])
    #                 r_end = int(stim_range[j].split("-")[1]) + 1
    #                 all_sweeps = list(range(r_start, r_end))
    #                 stim_sweeps.extend(all_sweeps)
    #             else:
    #                 stim_sweeps.append(int(stim_range[j][0]))

    #         stim_sweeps_VC = self.raw_df[
    #             self.raw_df.columns.intersection(set(stim_sweeps))
    #         ]
    #         sweeps_dict[stim_name] = stim_sweeps_VC

    #     # drop keys with empty dataframes
    #     self.sweeps_dict = {
    #         k: v for (k, v) in sweeps_dict.items() if not v.empty
    #     }

    def filter_traces(self, traces):
        """
        add filtered traces attrbute to data object
        lowpass_freq: frequency in Hz to pass to elephant filter
        """
        traces_filtered = elephant.signal_processing.butter(
            traces.T, lowpass_freq=500, fs=self.fs * 1000
        )
        traces_filtered = pd.DataFrame(traces_filtered).T

        return traces_filtered

    def calculate_mean_baseline(self, data):
        """
        Find the mean baseline in a given time series, defined the 100-450 ms window before stimulus onset
        Parameters
        ----------
        data: pandas.Series or pandas.DataFrame
            The time series data for which you want a baseline.
        baseline_start: int or float
            The time in ms when baseline starts.
        baseline_end: int or float
            The length of the sweep and the time when baseline ends.

        Returns
        -------
        baseline: float or pandas.Series
            The mean baseline over the defined window
        """
        start = self.baseline_start * self.fs
        stop = self.baseline_end * self.fs
        window = data.iloc[start:stop]
        baseline = window.mean()

        return baseline

    def calculate_std_baseline(self, data):
        """
        Find the mean baseline in a given time series
        Parameters
        ----------
        data: pandas.Series or pandas.DataFrame
            The time series data for which you want a baseline.
        baseline_start: int or float
            The time in ms when baseline starts.
        baseline_end: int or float
            The length of the sweep and the time when baseline ends.

        Returns
        -------
        baseline: float or pandas.Series
            The mean baseline over the defined window
        """
        start = self.baseline_start * self.fs
        stop = self.baseline_end * self.fs
        window = data.iloc[start:stop]
        std = window.std()

        return std

    def calculate_mean_trace_peak(
        self, data, baseline, polarity="-", index=False
    ):
        """
        Find the peak EPSC value for a pandas.Series or for each sweep (column) of
        a pandas.DataFrame. This finds the absolute peak value of mean baseline
        subtracted data.

        Parameters:
        -----------

        data: pandas.Series or pandas.DataFrame
            Time series data with stimulated synaptic response triggered at the
            same time for each sweep.
        baseline: scalar or pandas.Series
            Mean baseline values used to subtract for each sweep.
        polarity: str
            The expected polarity of the EPSC; negative: '-'; postitive: '+'.
            Default is '-'.
        post_stim: int or float
            Time in ms that marks the end of the sampling window post stimulus.
            Default is 100 ms.
        index: bool
            Determines whether or not to return the peak index in addition to the peak.

        Returns
        -------
        epsc_peaks: pandas.Series
            The absolute peak of mean baseline subtracted time series data.
        epsc_peak_index: int
            The time at which the peak occurs
        peak_window: pandas.DataFrame
            The window of the time series data where the peak is identified
        """

        subtracted_data = data - baseline
        start = self.stim_time * self.fs
        end = (self.stim_time + self.post_stim) * self.fs
        peak_window = subtracted_data.iloc[start:end]

        if index is True:
            if polarity == "-":
                epsc_peaks = peak_window.min()
                epsc_peaks_index = peak_window.idxmin()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
                epsc_peaks_index = peak_window.idxmax()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks, epsc_peaks_index, peak_window

        elif index is False:
            if polarity == "-":
                epsc_peaks = peak_window.min()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks, peak_window

    def calculate_event_peak(
        self, data, pos, baseline, polarity="-", index=False
    ):
        """
        Find the peak EPSC value for a pandas.Series or for each sweep (column) of
        a pandas.DataFrame. This finds the absolute peak value of mean baseline
        subtracted data.

        Parameters:
        -----------

        data: pandas.Series or pandas.DataFrame
            Time series data with stimulated synaptic response triggered at the
            same time for each sweep.
        baseline: scalar or pandas.Series
            Mean baseline values used to subtract for each sweep.
        mod_pos: scalar or pandas.Series
            The position of the event determined by MOD, at the original fs.
        polarity: str
            The expected polarity of the EPSC; negative: '-'; postitive: '+'.
            Default is '-'.
        post_stim: int or float
            Time in ms that marks the end of the sampling window post stimulus.
            Default is 100 ms.
        index: bool
            Determines whether or not to return the peak index in addition to the peak.

        Returns
        -------
        epsc_peaks: pandas.Series
            The absolute peak of mean baseline subtracted time series data.
        epsc_peak_index: int
            The time at which the peak occurs
        peak_window: pandas.DataFrame
            The window of the time series data where the peak is identified
        """

        subtracted_data = data - baseline
        start = pos - 50
        end = pos + 50

        peak_window = subtracted_data.iloc[start:end]

        if index is True:
            if polarity == "-":
                epsc_peaks = peak_window.min()
                epsc_peaks_index = peak_window.idxmin()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
                epsc_peaks_index = peak_window.idxmax()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks, epsc_peaks_index, peak_window

        elif index is False:
            if polarity == "-":
                epsc_peaks = peak_window.min()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks, peak_window

    def calculate_latency_jitter(self, window, epsc_peaks, mean_trace=False):
        """
        Finds the latency of response onset and the jitter of latency.
        Parameters
        ----------
        window: pandas.Series or pandas.DataFrame
            The time series data window in which the peak is found.
        epsc_peaks: float or pandas.Series
            The absolute peak of mean baseline subtracted time series data.
        mean_trace: bool
            Determines whether or not to return jitter in addition to latency.

        Returns
        -------
        onset: float or pandas.Series
            The absolute time of response onset
        latency: float or pandas.Series
            The latency of response onset from stim time, defined as 5% of
            epsc_peaks
        jitter: float or pandas.Series
            The std of latency to response onset (if calculated on multiple
            traces)
        """
        onset_amp = epsc_peaks * 0.05
        latency = []
        onset = []
        for sweep in range(len(window.columns)):
            sweep_window = window.iloc[:, sweep]
            onset_idx = np.argmax(sweep_window <= onset_amp[sweep])
            onset_time = sweep_window.index[onset_idx]
            onset.append(onset_time)
            sweep_latency = onset_time - self.stim_time
            latency.append(sweep_latency)

        if mean_trace is False:
            jitter = np.std(latency)
            return onset, latency, jitter

        elif mean_trace is True:
            return onset, latency, None

    def calculate_timetopeak(self, window, response_onset):
        """
        Find the mean baseline in a given time series
        Parameters
        ----------
        window: pandas.Series or pandas.DataFrame
            The time series data window in which the peak is found.
        response_onset: int or float
            Time in ms at which response onset occurs.

        Returns
        -------
        time_to_peak: float or pandas.Series
            The time to peak from response onset, in ms

        """
        peak_time = window.idxmin()
        time_to_peak = peak_time - response_onset

        return time_to_peak

    def calculate_responses(
        self, baseline_std, peak_mean, timetopeak, threshold=None
    ):
        """
            Decides on whether there is a response above 2x, 3x above the baseline std,
            or a user-selectable cutoff.
            Parameters
            ----------
            baseline_std: int or float
                The std of the baseline of the mean filtered trace.
            peak_mean: int or float
                The current peak of the mean filtered trace.
            timetopeak: int or float
                The time to current peak. Usually uses the mean trace time to peak
            threshold: int, float (optional)
                If supplied, will provide another threshold in addition to the 2x and 3x
                above the baseline std to threshold the response checker.
            Returns
            -------
            responses: pd.DataFrame(bool)
                A DataFrame with bool for responses above the threshold in the column header.
            """
        # takes values out of series format to enable boolean comparison
        baseline_std = baseline_std[0]
        peak_mean = peak_mean[0]

        response_2x = abs(peak_mean) > baseline_std * 2  # and timetopeak < 10
        response_3x = abs(peak_mean) > baseline_std * 3  # and timetopeak < 10

        if threshold is None:
            responses = pd.DataFrame(
                {
                    "Response 2x STD": response_2x,
                    "Response 3x STD": response_3x,
                },
                index=range(1),
            )
        else:
            response_threshold = abs(peak_mean) > baseline_std * threshold
            response_string = "Response {}x STD".format(threshold)

            responses = pd.DataFrame(
                {
                    "Response 2x STD": response_2x,
                    "Response 3x STD": response_3x,
                    response_string: response_threshold,
                },
                index=range(1),
            )

        return responses

    # def extract_drug_sweeps(self):
    #     """
    #     Puts baseline-subtracted NBQX wash-in traces into a df
    #     """
    #     sweeps_dict = self.drug_sweeps_dict
    #     traces = sweeps_dict["NBQX wash-in"]
    #     # mean_trace = traces.mean(axis=1)

    #     # filter traces
    #     traces_filtered = self.filter_traces(traces)
    #     mean_trace_filtered = pd.DataFrame(traces_filtered.mean(axis=1))

    #     # convert time to ms
    #     time = np.arange(0, len(traces) / FS, 1 / FS)
    #     mean_trace_filtered.index = time
    #     traces_filtered.index = time

    #     # find mean baseline, defined as the last 3s of the sweep
    #     baseline = self.calculate_mean_baseline(
    #         traces_filtered, baseline_start=100, baseline_end=450
    #     )
    #     mean_baseline = self.calculate_mean_baseline(
    #         mean_trace_filtered, baseline_start=100, baseline_end=450
    #     )

    #     # calculates the baseline-subtracted mean trace for plotting purposes
    #     sub_mean_trace = mean_trace_filtered - mean_baseline

    #     return sub_mean_trace

    # def extract_VC_sweep(self, selected_condition, sweep_number):
    #     """
    #     Extracts a single VC sweep for plotting onset latency. Select the
    #     sweep with sweep_number, which is the sweep # within a given set of
    #     stim sweeps.
    #     """
    #     # if self.cell_name == "JH20210923_c2":
    #     #     selected_condition = "50%, 1 ms"
    #     # else:
    #     #     selected_condition = ",".join(FileSettings.SELECTED_CONDITION)

    #     selected_sweep = self.filtered_traces_dict[selected_condition][
    #         sweep_number
    #     ]

    #     return selected_sweep

    # def extract_FI_sweep(self, sweep_number):
    #     """
    #     Extracts a single FI step sweep for plotting STC spikes. Select the
    #     sweep with sweep_number
    #     """
    #     sweeps_dict = self.ic_sweeps_dict
    #     traces = sweeps_dict["FI current steps"]

    #     # convert time to ms
    #     time = np.arange(0, len(traces) / FS, 1 / FS)
    #     traces.index = time

    #     # filter traces
    #     traces_filtered = self.filter_traces(traces)
    #     mean_trace_filtered = pd.DataFrame(traces_filtered.mean(axis=1))

    #     mean_trace_filtered.index = time
    #     traces_filtered.index = time
    #     traces_filtered.columns = traces.columns

    #     extracted_sweep = traces[sweep_number]

    #     # # find mean baseline, defined as the last 3s of the sweep
    #     # baseline = self.calculate_mean_baseline(
    #     #     traces_filtered, baseline_start=100, baseline_end=450
    #     # )
    #     # mean_baseline = self.calculate_mean_baseline(
    #     #     mean_trace_filtered, baseline_start=100, baseline_end=450
    #     # )

    #     return extracted_sweep

    def calculate_event_stats(self):
        traces = self.traces
        self.traces_filtered = self.filter_traces(traces)

        self.time = np.arange(0, len(traces) / self.fs, 1 / self.fs)
        self.traces_filtered.index = self.time

        # find baseline, defined as the last 3s of the sweep
        baseline = self.calculate_mean_baseline(self.traces_filtered)
        std_baseline = self.calculate_std_baseline(self.traces_filtered)

        # subtract mean baseline from all filtered traces - this is for
        # plotting individual traces
        self.traces_filtered_sub = self.traces_filtered - baseline

        # finds the new positions and peaks of identified events using MOD file
        new_amps = []
        new_pos_list = []

        for index, row in self.mod_events_df.iterrows():
            sweep = int(row["Sweep"])
            pos = int(row["Subtracted pos"])
            sweep_baseline = baseline[sweep]

            event_peak, new_pos, peak_window = self.calculate_event_peak(
                self.traces_filtered[sweep], pos, sweep_baseline, index=True
            )

            new_amps.append(event_peak)
            new_pos_list.append(new_pos)

        new_amps = pd.DataFrame(new_amps, columns=["New amplitude (pA)"])
        new_pos_list = pd.DataFrame(new_pos_list, columns=["New pos"])

        event_stats = pd.concat(
            [self.mod_events_df, new_pos_list, new_amps], axis=1
        )

        self.event_stats = event_stats

        # self.plot_events()

        # pdb.set_trace

    def calculate_mean_trace_stats(self):

        # filter traces
        mean_trace_filtered = pd.DataFrame(self.traces_filtered.mean(axis=1))
        mean_trace_filtered.index = self.time

        # find mean baseline, defined as the last 3s of the sweep
        mean_baseline = self.calculate_mean_baseline(mean_trace_filtered)

        # find std of baseline
        mean_std_baseline = self.calculate_std_baseline(mean_trace_filtered)

        # calculates the baseline-subtracted mean trace for plotting purposes
        self.sub_mean_trace = mean_trace_filtered - mean_baseline

        # find current peak
        mean_trace_peak, mean_peak_window = self.calculate_mean_trace_peak(
            mean_trace_filtered, mean_baseline, polarity="-"
        )

        # find latency to response onset and jitter, in ms
        # onset, latency, jitter = self.calculate_latency_jitter(
        #     peak_window, current_peaks, mean_trace=False
        # )
        (
            mean_trace_onset,
            mean_trace_latency,
            mean_trace_jitter,
        ) = self.calculate_latency_jitter(
            mean_peak_window, mean_trace_peak, mean_trace=True
        )

        # latency_mean = np.asarray(latency).mean()

        # # find time to peak, in ms
        # time_to_peak = self.calculate_timetopeak(peak_window, onset)
        # mean_trace_time_to_peak = self.calculate_timetopeak(
        #     mean_peak_window, mean_trace_onset
        # )
        # time_to_peak_mean = time_to_peak.mean()

        # # determines whether the cell is responding, using mean_trace_filtered
        # responses = self.calculate_responses(
        #     mean_std_baseline, mean_trace_peak, mean_trace_time_to_peak[0]
        # )

        # # collects measurements into cell dict, nested dict for each stim condition
        # mean_trace_dict = {
        #     "Cell name": self.cell_name,
        #     "Dataset": self.dataset,
        #     "Cell Type": self.cell_type,
        #     "Mean Trace Peak (pA)": mean_trace_peak[0],
        #     "Mean Trace Onset Latency (ms)": mean_trace_latency[0],
        #     "Mean Trace Time to Peak (ms)": mean_trace_time_to_peak[0],
        #     "Response 2x STD": responses["Response 2x STD"][0],
        #     "Response 3x STD": responses["Response 3x STD"][0],
        # }
        # print(pd.DataFrame(mean_trace_dict, index=[0]))

        # stim_dict = {
        #     "Cell name": self.cell_name,
        #     "Dataset": self.dataset,
        #     "Cell Type": self.cell_type,
        #     "Raw Peaks (pA)": current_peaks.tolist(),
        #     "Mean Raw Peaks (pA)": current_peaks_mean,
        #     "Mean Trace Peak (pA)": mean_trace_peak[0],
        #     "Onset Times (ms)": onset,
        #     "Onset Latencies (ms)": latency,
        #     "Onset Jitter": jitter,
        #     "Mean Onset Latency (ms)": latency_mean,
        #     "Onset SEM": sem(latency),
        #     "Mean Trace Onset Latency (ms)": mean_trace_latency[0],
        #     "Time to Peaks (ms)": time_to_peak.tolist(),
        #     "Mean Time to Peak (ms)": time_to_peak_mean,
        #     "Time to Peak SEM": sem(time_to_peak),
        #     "Mean Trace Time to Peak (ms)": mean_trace_time_to_peak[0],
        #     "Response 2x STD": responses["Response 2x STD"][0],
        #     "Response 3x STD": responses["Response 3x STD"][0],
        # }

        # return (
        #     self.traces_filtered_sub,
        #     sub_mean_trace,
        #     current_peaks,
        #     stim_dict,
        # )

    def plot_events(self):
        """
        Sanity check; plots all the individual sweeps with identified event 
        peaks marked. By default, all traces are hidden and only shown when
        selected in figure legends.
        """
        events_fig = go.Figure()

        for sweep in range(len(self.traces_filtered_sub.columns)):

            sweep_events_pos = self.event_stats.loc[
                self.event_stats["Sweep"] == sweep
            ]["New pos"].values
            sweep_events_amp = self.event_stats.loc[
                self.event_stats["Sweep"] == sweep
            ]["New amplitude (pA)"].values

            # pdb.set_trace()

            window_toplot = self.traces_filtered_sub[sweep][
                (self.tp_start + self.tp_length) : :
            ]

            events_fig.add_trace(
                go.Scatter(
                    x=window_toplot.index,
                    y=window_toplot,
                    mode="lines",
                    name="sweep {}".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            events_fig.add_trace(
                go.Scatter(
                    x=sweep_events_pos,
                    y=sweep_events_amp,
                    mode="markers",
                    name="sweep {} peaks".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

        events_fig.update_xaxes(title_text="Time (ms)")
        events_fig.update_yaxes(title_text="Amplitude (pA)")

        events_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        events_fig.show()
        self.events_fig = events_fig

    def plot_event_psth(self):
        """
        Makes raster plot of all identified events for each sweep.
        """
        raster_df = self.event_stats[["Sweep", "New pos"]]

        # make sweep numbers go from 1-30 instead of 0-29
        new_sweeps = raster_df["Sweep"].cat.codes + 1

        # sets background color to white
        layout = go.Layout(plot_bgcolor="rgba(0,0,0,0)",)

        # make overall fig layout
        psth_fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.025,
            x_title="Time (ms)",
            shared_xaxes=True,
        )

        # add raster plot
        psth_fig.add_trace(
            go.Scatter(
                x=raster_df["New pos"],
                y=new_sweeps,
                mode="markers",
                marker=dict(symbol="line-ns", line_width=1, size=10),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        psth_fig.update_yaxes(
            title_text="Trial",
            row=1,
            col=1,
            tickvals=[1, len(self.traces_filtered_sub.columns)],
            showgrid=False,
            zeroline=False,
        )

        psth_fig.update_xaxes(
            row=1, col=1, showticklabels=False, showgrid=False
        )

        psth_fig.update_xaxes(
            # title_text="Time (ms)",
            range=[(self.tp_start + self.tp_length), 6020],
        )

        # make PSTH
        psth_df = raster_df["New pos"]

        # do frequency in Hz, # of events per second, divided by # of sweeps
        counts, bins = np.histogram(psth_df, bins=range(0, 6030, 10))
        # this puts bar in between the edges of the bin
        bar_bins = 0.5 * (bins[:-1] + bins[1:])
        freq = counts / len(self.traces_filtered_sub.columns) / 1e-2

        psth_fig.add_trace(
            go.Bar(
                x=bar_bins,
                y=freq,
                showlegend=False,
                marker=dict(color="#D39DDD"),
            ),
            row=2,
            col=1,
        )

        # this removes the white outline of the bar graph to emulate histogram
        psth_fig.update_traces(marker=dict(line=dict(width=0)), row=2, col=1)

        psth_fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        psth_fig.update_layout(bargap=0)

        # adds blue overlay to show light stim duration
        psth_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
            row="all",
            col=1,
        )

        # add main title, x-axis titles
        psth_fig.update_layout(
            title_text="{}, {} PSTH".format(self.cell_name, self.cell_type),
            title_x=0.5,
        )

        # psth_fig.show()

        # calculate isi, put raster_df into event "spike train"

        # event_trains = pd.DataFrame()
        # isi_df = pd.DataFrame()
        # instantaneous_FR = pd.DataFrame()

        # for sweep in range(len(self.traces_filtered_sub.columns)):
        #     # makes event train
        #     sweep_train = []
        #     events = raster_df.loc[raster_df["Sweep"] == sweep][
        #         "New pos"
        #     ].tolist()
        #     events_train = SpikeTrain(events * s, t_stop=6020)
        #     sweep_train.append(events_train)
        #     event_trains[sweep] = sweep_train

        #     # calculates isi
        #     isi = elephant.statistics.isi(sweep_train)
        #     isi = isi.tolist()
        #     isi_df[sweep] = isi

        #     # calculates instantaneous FR

        #     # unclear what sampling_period should be.. time stamp resolution of
        #     # the spike times. inverse of sampling rate, is this 1/FS?  would
        #     # take too long

        #     # with sampling_period = 10*ms, this leads to 602000 samples, i.e.
        #     # one sample/FR per 100 ms of the sweep

        #     # can't use gaussian/kernel because uneven distribution
        #     rate = elephant.statistics.instantaneous_rate(
        #         events_train, sampling_period=1000 * ms, kernel="auto"
        #     )
        #     instantaneous_FR[sweep] = rate.magnitude.tolist()

        # event_trains = event_trains.T
        # isi_df = isi_df.T

        #

        # gets isi for each sweep
        # how to put the isi back into time?

        # testing bayesian smoothing,
        # code from https://gist.github.com/AustinRochford/d640a240af12f6869a7b9b592485ca15
        # but changed to work with updated pymc

        # below is my data

        # N_KNOT = len(freq)
        # knots = freq
        # c = np.random.normal(size=N_KNOT)  # what are these spline coefficient?

        # x = bar_bins  # is this right? using the midpoint of PSTH bins
        # y = freq
        # # or can I find k and coefficients myself here
        # knots, c, k = scipy.interpolate.splrep(x=bar_bins, y=freq)
        # # uses k=3 degrees for B-spline
        # N_KNOT = len(knots)

        # spline = scipy.interpolate.BSpline(knots, c, k, extrapolate=False)
        # y = spline(x)

        # fig = go.Figure()
        # fig.add_trace(
        #     go.Scatter(x=bins, y=spline(bins), name="spline")
        # )  # plots splines
        # fig.add_trace(
        #     go.Scatter(x=bar_bins, y=y, mode="markers", name="freq")
        # )  # plots freq points

        # fig.show()

        # pdb.set_trace()

        N_KNOT = 10  # arbitrary # of knots
        # quantiles = np.linspace(
        #     0, 1, N_KNOT
        # )  # if I want to use quantiles as knots
        knots = np.linspace(0, bins[-1], N_KNOT)  # interior knots
        # c = np.random.normal(size=N_KNOT)

        # feed interior knots to get spline coefficients

        knots, c, k = scipy.interpolate.splrep(
            x=bar_bins, y=freq, task=-1, t=knots[1:-1]
        )

        # F = scipy.interpolate.PPoly.from_spline(tck)

        spline = scipy.interpolate.BSpline(knots, c, 3, extrapolate=False)

        x = bar_bins
        y = freq
        x_plot = bins

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=x_plot, y=spline(x_plot), name="spline")
        )  # plots splines
        fig.add_trace(
            go.Scatter(x=bar_bins, y=y, mode="markers", name="freq")
        )  # plots freq points
        fig.show()

        # pdb.set_trace()

        N_MODEL_KNOTS = 5 * N_KNOT

        # N_MODEL_KNOTS = N_KNOT
        model_knots = np.linspace(0, bins[-1], N_MODEL_KNOTS)

        # running model

        basis_funcs = scipy.interpolate.BSpline(
            model_knots, np.eye(N_MODEL_KNOTS), k=3
        )

        Bx = basis_funcs(x)

        Bx_ = shared(Bx)

        with pm.Model() as model:
            σ_a = pm.HalfCauchy("σ_a", 5.0)
            a0 = pm.Normal("a0", 0.0, 10.0)
            Δ_a = pm.Normal("Δ_a", 0.0, 1.0, shape=N_MODEL_KNOTS)
            a = pm.Deterministic("a", a0 + (σ_a * Δ_a).cumsum())

            σ = pm.HalfCauchy("σ", 5.0)

            obs = pm.Normal("obs", Bx_.dot(a), σ, observed=y)

        with model:
            trace = pm.sample(target_accept=0.95)

        pm.energyplot(trace)

        Bx_.set_value(basis_funcs(bins))

        with model:
            pp_trace = pm.sample_posterior_predictive(trace, 1000)

        fig.add_trace(
            go.Scatter(
                x=bins, y=pp_trace["obs"].mean(axis=0), name="spline estimate"
            )
        )
        fig.show()

        pdb.set_trace()

        # below is example code parameters

        """
        N_KNOT = 30

        knots = np.linspace(-0.5, 1.5, N_KNOT)
        c = np.random.normal(size=N_KNOT)
        spline = scipy.interpolate.BSpline(knots, c, 3, extrapolate=False)

        x = np.random.uniform(0, 1, 100)
        x.sort()
        y = spline(x) + np.random.normal(scale=0.25, size=x.size)
        x_plot = np.linspace(0, 1, 100)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x_plot, y=spline(x_plot), name="spline")
        )  # plots splines
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name="freq")
        )  # plots freq points
        fig.show()

        N_MODEL_KNOTS = 5 * N_KNOT
        model_knots = np.linspace(-0.5, 1.5, N_MODEL_KNOTS)

        # running model

        basis_funcs = scipy.interpolate.BSpline(
            knots, np.eye(N_MODEL_KNOTS), k=3
        )

        Bx = basis_funcs(x)

        Bx_ = shared(Bx)

        with pm.Model() as model:
            σ_a = pm.HalfCauchy("σ_a", 5.0)
            a0 = pm.Normal("a0", 0.0, 10.0)
            Δ_a = pm.Normal("Δ_a", 0.0, 1.0, shape=N_MODEL_KNOTS)
            a = pm.Deterministic("a", a0 + (σ_a * Δ_a).cumsum())

            σ = pm.HalfCauchy("σ", 5.0)

            obs = pm.Normal("obs", Bx_.dot(a), σ, observed=y)

        with model:
            trace = pm.sample(target_accept=0.95)

        pm.energyplot(trace)

        Bx_.set_value(basis_funcs(x_plot))

        with model:
            pp_trace = pm.sample_posterior_predictive(trace, 1000)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=x_plot, y=spline(x_plot), name="spline")
        )  # plots splines
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name="freq")
        )  # plots freq points
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=pp_trace["obs"].mean(axis=0),
                name="spline estimate",
            )
        )
        fig.show()
        """

        pdb.set_trace()

    # def plot_counts_psth(self):
    #     """
    #     Makes a PSTH of events identified across all sweeps, using raster_df.
    #     Not currently used since we want frequency, which needs bar graph in
    #     plotly.
    #     """

    #     # use bin width of 10 ms, Burton 2015 paper
    #     bin_intervals = np.arange(0, 6030, 10)

    #     # assign bin numbers based on event positions
    #     psth_df = raster_df["New pos"]
    #     # psth_cut = pd.cut(
    #     #     raster_df["New pos"],
    #     #     bin_intervals,
    #     #     include_lowest=True,
    #     #     right=False,
    #     # )

    #     # # get counts for each bin
    #     # counts = pd.value_counts(psth_df)

    #     psth_fig = go.Figure()
    #     psth_fig.add_trace(
    #         go.Histogram(
    #             x=psth_df,
    #             xbins=dict(
    #                 start=bin_intervals[0], size=10, end=bin_intervals[-1]
    #             ),
    #             # histnorm="probability",
    #         )
    #     )

    #     psth_fig.add_vrect(
    #         x0=self.stim_time,
    #         x1=self.stim_time + 100,
    #         fillcolor="#33F7FF",
    #         opacity=0.5,
    #         layer="below",
    #         line_width=0,
    #     )

    #     psth_fig.update_xaxes(
    #         title_text="Time (ms)",
    #         range=[(self.tp_start + self.tp_length), 6020],
    #     )
    #     psth_fig.update_yaxes(title_text="Counts")
    #     # psth_fig.show()

    #     pdb.set_trace()

    def plot_mean_trace(self):
        """
        Plots the averaged trace ontop of all individual sweeps. Individual 
        sweeps are hidden by default unless selected to show with legend.
        """

        mean_trace_fig = go.Figure()

        # plots individual sweeps
        for sweep in range(len(self.traces_filtered_sub.columns)):
            indiv_toplot = self.traces_filtered_sub[sweep][
                (self.tp_start + self.tp_length) : :
            ]

            mean_trace_fig.add_trace(
                go.Scatter(
                    x=indiv_toplot.index,
                    y=indiv_toplot,
                    mode="lines",
                    marker=dict(color="gray", line_width=1),
                    name="sweep {}".format(sweep),
                    legendgroup="indiv_sweeps",
                    visible="legendonly",
                )
            )

        # plots averaged trace second so it shows up on top
        window_toplot = self.sub_mean_trace[
            (self.tp_start + self.tp_length) : :
        ]
        mean_trace_fig.add_trace(
            go.Scatter(
                x=window_toplot.index,
                y=window_toplot.squeeze(),
                mode="lines",
                name="Averaged sweep",
                legendgroup="avg_sweep",
                # visible="legendonly",
            )
        )

        mean_trace_fig.update_xaxes(title_text="Time (ms)")
        mean_trace_fig.update_yaxes(title_text="Amplitude (pA)")

        mean_trace_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        mean_trace_fig.show()
        pdb.set_trace()
        # self.events_fig = events_fig

    def get_mod_events(self):
        mod_events = io.loadmat(
            "/home/jhuang/Documents/phd_projects/mod-2.0/output/JH200313_c3_light100.mod.w4.e1.h13.minidet.mat"
        )
        pos_list = mod_events["pos"]
        flat_pos = pd.DataFrame(
            [pos for sublist in pos_list for pos in sublist]
        )

        num_sweeps = len(self.raw_df.columns)
        raw_intervals = np.arange(0, (num_sweeps + 1) * 150500, 150500)

        # assign sweep numbers based on event positions
        sweep_pos = pd.cut(
            flat_pos[0],
            raw_intervals,
            labels=np.arange(0, num_sweeps),
            include_lowest=True,
            right=False,
        )

        mod_events_df = pd.DataFrame()
        mod_events_df["Sweep"] = sweep_pos
        mod_events_df["Raw pos"] = flat_pos

        sweep_array = np.asarray(mod_events_df["Sweep"])

        # get absolute time of event within each sweep
        subtract_time = sweep_array * self.fs * 6020
        mod_events_df["Subtracted pos"] = flat_pos.squeeze() - subtract_time
        mod_events_df["Event pos (ms)"] = (
            mod_events_df["Subtracted pos"] / self.fs
        )

        # amplitudes from MOD
        mod_events_df["MOD amplitude (pA)"] = mod_events["amplitude"]

        self.mod_events_df = mod_events_df

    def make_cell_analysis_dict(self):

        # create dict to hold analysis results for each stim set
        filtered_traces_dict = {}
        cell_analysis_dict = {}
        power_curve_df = pd.DataFrame()
        all_mean_traces = pd.DataFrame()
        # pdb.set_trace()
        for stim_id in range(len(list(self.sweeps_dict))):
            (
                stim_condition,
                traces_filtered_sub,
                sub_mean_trace,
                current_peaks,
                stim_dict,
            ) = self.calculate_stim_stats(stim_id)
            cell_analysis_dict[stim_condition] = stim_dict
            filtered_traces_dict[stim_condition] = traces_filtered_sub

            # collects peaks into power_curve_df, column for each stim condition
            stim_peaks = pd.DataFrame(current_peaks)
            stim_peaks.columns = [stim_condition]
            power_curve_df = pd.concat([power_curve_df, stim_peaks], axis=1)

            # collects mean traces into all_mean_traces, column for each stim condition
            mean_trace = pd.DataFrame(sub_mean_trace)
            mean_trace.columns = [stim_condition]
            all_mean_traces = pd.concat([all_mean_traces, mean_trace], axis=1)

        self.cell_analysis_dict = cell_analysis_dict
        self.power_curve_df = power_curve_df
        self.all_mean_traces = all_mean_traces
        self.filtered_traces_dict = filtered_traces_dict

    # # export analysis values to csv

    # def make_power_curve_stats_df(self):

    #     """
    #     The below makes power curve stats table used to plot power curve
    #     """
    #     power_curve_df = self.power_curve_df

    #     # how to convert column names to tuples, then can just pass tuples through to multiindex
    #     tuples = [
    #         tuple(condition.split(",")) for condition in power_curve_df.columns
    #     ]
    #     power_curve_df.columns = pd.MultiIndex.from_tuples(tuples)
    #     power_curve_df.index.names = ["Sweep"]
    #     power_curve_df = power_curve_df.T

    #     # get mean response and SEM for plotting
    #     power_curve_stats = pd.DataFrame()
    #     power_curve_stats[
    #         "Mean Response Amplitude (pA)"
    #     ] = power_curve_df.mean(axis=1)
    #     power_curve_stats["SEM"] = power_curve_df.sem(axis=1)
    #     power_curve_df = pd.concat(
    #         [power_curve_df, power_curve_stats], axis=1
    #     )  # for output to csv

    #     power_curve_stats.reset_index(inplace=True)
    #     power_curve_stats = power_curve_stats.rename(
    #         columns={"level_0": "Light Intensity", "level_1": "Light Duration"}
    #     )

    #     self.tuples = tuples
    #     self.power_curve_stats = power_curve_stats

    # def make_stats_df(self):

    #     """
    #     The below makes response stats tables used to plot graphs
    #     """
    #     cell_analysis_df = pd.DataFrame(self.cell_analysis_dict).T
    #     cell_analysis_df.index = pd.MultiIndex.from_tuples(self.tuples)

    #     cell_analysis_df = cell_analysis_df[
    #         [
    #             "Cell name",
    #             "Dataset",
    #             "Genotype",
    #             "Raw Peaks (pA)",
    #             "Mean Raw Peaks (pA)",
    #             "Mean Trace Peak (pA)",
    #             "Onset Latencies (ms)",
    #             "Mean Onset Latency (ms)",
    #             "Onset SEM",
    #             "Onset Jitter",
    #             "Mean Trace Onset Latency (ms)",
    #             "Time to Peaks (ms)",
    #             "Mean Time to Peak (ms)",
    #             "Time to Peak SEM",
    #             "Mean Trace Time to Peak (ms)",
    #             "Response 2x STD",
    #             "Response 3x STD",
    #         ]
    #     ]

    #     sweep_analysis_values = cell_analysis_df[
    #         ["Onset Latencies (ms)", "Time to Peaks (ms)"]
    #     ].copy()
    #     sweep_analysis_values = sweep_analysis_values.explode(
    #         ["Onset Latencies (ms)", "Time to Peaks (ms)"]
    #     )

    #     sweep_analysis_values.reset_index(inplace=True)
    #     sweep_analysis_values = sweep_analysis_values.rename(
    #         columns={"level_0": "Light Intensity", "level_1": "Light Duration"}
    #     )

    #     cell_analysis_df.reset_index(inplace=True)
    #     cell_analysis_df = cell_analysis_df.rename(
    #         columns={"level_0": "Light Intensity", "level_1": "Light Duration"}
    #     )

    #     self.sweep_analysis_values = sweep_analysis_values
    #     self.cell_analysis_df = cell_analysis_df

    #     # pdb.set_trace()

    # def export_stats_csv(self):
    #     """
    #     Exports sweep stats values (self.cell_analysis_df) to a csv file, not MultiIndex
    #     """

    #     stats_cleaned = self.cell_analysis_df.copy()
    #     stats_cleaned = stats_cleaned.drop(
    #         ["Raw Peaks (pA)", "Onset Latencies (ms)", "Time to Peaks (ms)",],
    #         axis=1,
    #     )

    #     base_path = os.path.join(
    #         "/home/jhuang/Documents/phd_projects/MMZ_STC_dataset/tables",
    #         self.dataset,
    #         self.genotype,
    #     )
    #     if not os.path.exists(base_path):
    #         os.makedirs(base_path)

    #     csv_filename = "{}_response_stats.csv".format(self.cell_name)
    #     path = os.path.join(base_path, csv_filename)
    #     stats_cleaned.to_csv(path, float_format="%8.4f", index=False)

    # def graph_curve_stats(self):
    #     """
    #     do a loop through available durations and intensities instead of hard
    #     coding. maybe need MultiIndex after all?? Put power curve + all stats
    #     measurements in subplots
    #     """

    #     power_curve_stats = self.power_curve_stats
    #     sweep_analysis_values = self.sweep_analysis_values
    #     cell_analysis_df = self.cell_analysis_df

    #     intensities = sweep_analysis_values["Light Intensity"].unique()
    #     durations = sweep_analysis_values["Light Duration"].unique()

    #     color = ["#0859C6", "#10A5F5", "#00DBFF"]
    #     curve_stats_fig = make_subplots(
    #         rows=3, cols=2, x_title="Light Intensity (%)"
    #     )

    #     # make the x-axis of light intensity to be used in each subplot

    #     x_sweep_dict = {}

    #     for duration in durations:
    #         x_sweep_intensity = sweep_analysis_values.loc[
    #             sweep_analysis_values["Light Duration"] == duration,
    #             ["Light Intensity"],
    #         ]

    #         x_sweep_dict[duration] = x_sweep_intensity

    #     max_key, max_value = max(
    #         x_sweep_dict.items(), key=lambda x: len(set(x[1]))
    #     )
    #     # pdb.set_trace()
    #     for count, duration in enumerate(durations):

    #         error = power_curve_stats.loc[
    #             power_curve_stats["Light Duration"] == duration, ["SEM"]
    #         ].squeeze()

    #         # if duration has only one intensity, resize_like the y
    #         # values for plotting
    #         # if x_dict[duration].nunique()[0] == 1:

    #         #     # onset latency
    #         #     y_onsetlatency = (
    #         #         sweep_analysis_values.loc[
    #         #             sweep_analysis_values["Light Duration"] == duration,
    #         #             ["Onset Latencies (ms)"],
    #         #         ]
    #         #         .reindex_like(bigger, method="ffill")
    #         #         .squeeze()
    #         #     )

    #         if len(intensities) > 1:
    #             if isinstance(error, float) != True:
    #                 # only make power curve if more than one intensity exists

    #                 # power curve
    #                 curve_stats_fig.add_trace(
    #                     go.Scatter(
    #                         x=power_curve_stats.loc[
    #                             power_curve_stats["Light Duration"]
    #                             == duration,
    #                             ["Light Intensity"],
    #                         ].squeeze(),
    #                         y=power_curve_stats.loc[
    #                             power_curve_stats["Light Duration"]
    #                             == duration,
    #                             ["Mean Response Amplitude (pA)"],
    #                         ].squeeze(),
    #                         name=duration,
    #                         error_y=dict(
    #                             type="data", array=error.values, visible=True
    #                         ),
    #                         mode="lines+markers",
    #                         line=dict(color=color[count]),
    #                         legendgroup=duration,
    #                     ),
    #                     row=1,
    #                     col=1,
    #                 )

    #         # buffering for durations with only one intensity
    #         if x_sweep_dict[duration].nunique()[0] == 1:

    #             sweep_partial = sweep_analysis_values.loc[
    #                 sweep_analysis_values["Light Duration"] == duration
    #             ].copy()

    #             # identifies the intensity present in the smaller duration
    #             partial_intens = sweep_partial["Light Intensity"].unique()[0]

    #             cell_partial = cell_analysis_df.loc[
    #                 cell_analysis_df["Light Duration"] == duration,
    #                 [
    #                     "Light Intensity",
    #                     "Light Duration",
    #                     "Onset Jitter",
    #                     "Mean Trace Onset Latency (ms)",
    #                     "Mean Trace Time to Peak (ms)",
    #                 ],
    #             ].copy()

    #             # pdb.set_trace()

    #             # if there exists a duration with more than 1 intensity,
    #             # use it as template for buffering
    #             if x_sweep_dict[max_key].nunique()[0] > 1:

    #                 sweep_template = sweep_analysis_values.loc[
    #                     sweep_analysis_values["Light Duration"] == max_key
    #                 ].copy()

    #                 sweep_template.loc[
    #                     sweep_template["Light Duration"] == max_key,
    #                     ["Onset Latencies (ms)", "Time to Peaks (ms)"],
    #                 ] = 0

    #                 sweep_template["Light Duration"] = duration

    #                 # for cell_analysis_df
    #                 cell_template = cell_analysis_df.loc[
    #                     cell_analysis_df["Light Duration"] == max_key,
    #                     [
    #                         "Light Intensity",
    #                         "Light Duration",
    #                         "Onset Jitter",
    #                         "Mean Trace Onset Latency (ms)",
    #                         "Mean Trace Time to Peak (ms)",
    #                     ],
    #                 ].copy()

    #                 cell_template.loc[
    #                     cell_template["Light Duration"] == max_key,
    #                     [
    #                         "Onset Jitter",
    #                         "Mean Trace Onset Latency (ms)",
    #                         "Mean Trace Time to Peak (ms)",
    #                     ],
    #                 ] = 0

    #                 cell_template["Light Duration"] = duration

    #                 x_cell_template = power_curve_stats.loc[
    #                     power_curve_stats["Light Duration"] == max_key,
    #                     ["Light Intensity"],
    #                 ]

    #             # if no duration exists with more than one intensity, make up
    #             # a skeletal five-intensity template for buffering
    #             else:
    #                 # creating list of intensities for template column
    #                 default_intensities = ["100%", "80%", "50%", "20%", "10%"]
    #                 intensities_template = [
    #                     intensity
    #                     for intensity in default_intensities
    #                     for i in range(10)
    #                 ]

    #                 # creating list of durations
    #                 durations_template = np.repeat(duration, 50)

    #                 # creating list of zeroes for y_sweep_values template
    #                 zeros_list = np.repeat(0, 50)

    #                 sweep_template = pd.DataFrame(
    #                     {
    #                         "Light Intensity": intensities_template,
    #                         "Light Duration": durations_template,
    #                         "Onset Latencies (ms)": zeros_list,
    #                         "Time to Peaks (ms)": zeros_list,
    #                     }
    #                 )

    #                 # account for cells where there are more than 10 sweeps
    #                 # for the partial_intens, need to add additional rows
    #                 # to sweep template
    #                 if len(sweep_partial) > 10:
    #                     repeat_n = len(sweep_partial) - 10
    #                     extra_row = pd.DataFrame(
    #                         {
    #                             "Light Intensity": partial_intens,
    #                             "Light Duration": duration,
    #                             "Onset Latencies (ms)": [0],
    #                             "Time to Peaks (ms)": [0],
    #                         }
    #                     )

    #                     last_idx = sweep_template.loc[
    #                         sweep_template["Light Intensity"] == partial_intens
    #                     ].index[-1]
    #                     insert_point = last_idx + 1
    #                     to_insert = extra_row.loc[
    #                         extra_row.index.repeat(repeat_n)
    #                     ]

    #                     sweep_template = pd.concat(
    #                         [
    #                             sweep_template.iloc[:insert_point],
    #                             to_insert,
    #                             sweep_template.iloc[insert_point:],
    #                         ]
    #                     )
    #                     sweep_template.reset_index(drop=True, inplace=True)

    #                 # creating skeletal template for y_cell_values
    #                 cell_template = pd.DataFrame(
    #                     {
    #                         "Light Intensity": default_intensities,
    #                         "Light Duration": np.repeat(duration, 5),
    #                         "Onset Jitter": np.repeat(0, 5),
    #                         "Mean Trace Onset Latency (ms)": np.repeat(0, 5),
    #                         "Mean Trace Time to Peak (ms)": np.repeat(0, 5),
    #                     }
    #                 )
    #                 # isolate the sweeps to slot into template
    #                 cell_partial = cell_analysis_df.loc[
    #                     cell_analysis_df["Light Duration"] == duration,
    #                     [
    #                         "Light Intensity",
    #                         "Light Duration",
    #                         "Onset Jitter",
    #                         "Mean Trace Onset Latency (ms)",
    #                         "Mean Trace Time to Peak (ms)",
    #                     ],
    #                 ].copy()

    #                 x_cell_template = pd.DataFrame(
    #                     {"Light Intensity": np.repeat(partial_intens, 5)}
    #                 )

    #             sweep_tobe_replaced = sweep_template.loc[
    #                 sweep_template["Light Intensity"] == partial_intens
    #             ]
    #             # pdb.set_trace()
    #             sweep_tobe_replaced.index = list(sweep_tobe_replaced.index)

    #             sweep_partial.set_index(
    #                 sweep_tobe_replaced.index[: len(sweep_partial.index)],
    #                 inplace=True,
    #             )

    #             sweep_template.update(sweep_partial)

    #             cell_tobe_replaced = cell_template.loc[
    #                 cell_template["Light Intensity"] == partial_intens
    #             ]

    #             cell_partial.set_index(cell_tobe_replaced.index, inplace=True)

    #             cell_template.update(cell_partial)

    #             y_sweep_values = sweep_template.copy()
    #             y_cell_values = cell_template.copy()
    #             x_cell_values = x_cell_template.copy()

    #         else:
    #             y_sweep_values = sweep_analysis_values.copy()
    #             y_cell_values = cell_analysis_df.copy()
    #             x_cell_values = power_curve_stats.loc[
    #                 power_curve_stats["Light Duration"] == duration,
    #                 ["Light Intensity"],
    #             ]
    #         # pdb.set_trace()
    #         # onset latency
    #         curve_stats_fig.add_trace(
    #             go.Box(
    #                 x=x_sweep_dict[duration].squeeze(),
    #                 y=y_sweep_values.loc[
    #                     y_sweep_values["Light Duration"] == duration,
    #                     ["Onset Latencies (ms)"],
    #                 ].squeeze(),
    #                 name=duration,
    #                 line=dict(color=color[count]),
    #                 legendgroup=duration,
    #             ),
    #             row=1,
    #             col=2,
    #         )

    #         # onset jitter
    #         curve_stats_fig.add_trace(
    #             go.Bar(
    #                 x=x_cell_values.squeeze(),
    #                 y=y_cell_values.loc[
    #                     y_cell_values["Light Duration"] == duration,
    #                     ["Onset Jitter"],
    #                 ].squeeze(),
    #                 name=duration,
    #                 marker_color=color[count],
    #                 legendgroup=duration,
    #             ),
    #             row=2,
    #             col=1,
    #         )

    #         # mean trace onset latency
    #         curve_stats_fig.add_trace(
    #             go.Bar(
    #                 x=x_cell_values.squeeze(),
    #                 y=y_cell_values.loc[
    #                     y_cell_values["Light Duration"] == duration,
    #                     ["Mean Trace Onset Latency (ms)"],
    #                 ].squeeze(),
    #                 name=duration,
    #                 marker_color=color[count],
    #                 legendgroup=duration,
    #             ),
    #             row=2,
    #             col=2,
    #         )

    #         # time to peak
    #         curve_stats_fig.add_trace(
    #             go.Box(
    #                 x=x_sweep_dict[duration].squeeze(),
    #                 y=y_sweep_values.loc[
    #                     y_sweep_values["Light Duration"] == duration,
    #                     ["Time to Peaks (ms)"],
    #                 ].squeeze(),
    #                 name=duration,
    #                 line=dict(color=color[count]),
    #                 legendgroup=duration,
    #             ),
    #             row=3,
    #             col=1,
    #         )

    #         # mean trace time to peak
    #         curve_stats_fig.add_trace(
    #             go.Bar(
    #                 x=x_cell_values.squeeze(),
    #                 y=y_cell_values.loc[
    #                     y_cell_values["Light Duration"] == duration,
    #                     ["Mean Trace Time to Peak (ms)"],
    #                 ].squeeze(),
    #                 name=duration,
    #                 marker_color=color[count],
    #                 legendgroup=duration,
    #             ),
    #             row=3,
    #             col=2,
    #         )

    #         # Update xaxis properties
    #         # curve_stats_fig.update_xaxes(autorange="reversed")
    #         # this defines the intensities order for x-axes
    #         curve_stats_fig.update_xaxes(
    #             categoryorder="array", categoryarray=np.flip(intensities)
    #         )

    #         # Update yaxis properties
    #         curve_stats_fig.update_yaxes(
    #             title_text="Response Amplitude (pA)",
    #             row=1,
    #             col=1,
    #             autorange="reversed",
    #         )
    #         curve_stats_fig.update_yaxes(
    #             title_text="Onset Latency (ms)", row=1, col=2
    #         )
    #         curve_stats_fig.update_yaxes(
    #             title_text="Onset Jitter", row=2, col=1
    #         )
    #         curve_stats_fig.update_yaxes(
    #             title_text="Mean Trace Onset Latency (ms)", row=2, col=2
    #         )
    #         curve_stats_fig.update_yaxes(
    #             title_text="Time to Peak (ms)", row=3, col=1
    #         )
    #         curve_stats_fig.update_yaxes(
    #             title_text="Mean Trace Time to Peak (ms)", row=3, col=2
    #         )

    #     curve_stats_fig.update_layout(
    #         # yaxis_title='Onset Latency (ms)',
    #         boxmode="group"  # group together boxes of the different traces for each value of x
    #     )

    #     # below is code from stack overflow to hide duplicate legends
    #     names = set()
    #     curve_stats_fig.for_each_trace(
    #         lambda trace: trace.update(showlegend=False)
    #         if (trace.name in names)
    #         else names.add(trace.name)
    #     )

    #     curve_stats_fig.update_layout(legend_title_text="Light Duration")

    #     # curve_stats_fig.show()

    #     self.curve_stats_fig = curve_stats_fig

    # def make_mean_traces_df(self):

    #     """
    #     The below makes mean traces df used to plot graphs
    #     """

    #     mean_trace_df = self.all_mean_traces.T
    #     mean_trace_df.index = pd.MultiIndex.from_tuples(self.tuples)

    #     mean_trace_df.reset_index(inplace=True)
    #     mean_trace_df = mean_trace_df.rename(
    #         columns={"level_0": "Light Intensity", "level_1": "Light Duration"}
    #     )

    #     self.mean_trace_df = mean_trace_df

    #     return mean_trace_df

    # def graph_response_trace(self):
    #     """
    #     Plots the baseline-subtracted mean trace for each stimulus condition around the response time,
    #     one subplot for each duration, if applicable
    #     """

    #     # intensities and durations, and color might need to become self variables

    #     sweep_analysis_values = self.sweep_analysis_values
    #     intensities = sweep_analysis_values["Light Intensity"].unique()
    #     durations = sweep_analysis_values["Light Duration"].unique()

    #     # blue colors
    #     color = ["#0859C6", "#10A5F5", "#00DBFF"]

    #     stim_columns = self.mean_trace_df.loc[
    #         :, ["Light Intensity", "Light Duration"]
    #     ]
    #     traces_to_plot = self.mean_trace_df.loc[
    #         :, 500.00:700.00
    #     ]  # only plots first 400-1000 ms
    #     traces_to_plot_combined = pd.concat(
    #         [stim_columns, traces_to_plot], axis=1
    #     )

    #     mean_traces_fig = make_subplots(
    #         # rows=len(intensities), cols=1,
    #         rows=1,
    #         cols=len(intensities),
    #         subplot_titles=(intensities[::-1] + " Light Intensity"),
    #         shared_yaxes=True,
    #         x_title="Time (ms)",
    #         y_title="Amplitude (pA)",
    #     )

    #     # new method for hiding duplicate legends:
    #     # create a list to track each time a duration has been plotted, and only show legends
    #     # for the first time the duration is plotted
    #     duration_checker = []

    #     for intensity_count, intensity in enumerate(intensities):
    #         for duration_count, duration in enumerate(durations):

    #             # plot sweeps from all intensities of one duration
    #             y_toplot = traces_to_plot_combined.loc[
    #                 (traces_to_plot_combined["Light Intensity"] == intensity)
    #                 & (traces_to_plot_combined["Light Duration"] == duration),
    #                 500.00::,
    #             ].squeeze()
    #             mean_traces_fig.add_trace(
    #                 go.Scatter(
    #                     x=traces_to_plot.columns,
    #                     y=y_toplot,
    #                     name=duration,
    #                     mode="lines",
    #                     line=dict(color=color[duration_count]),
    #                     showlegend=False
    #                     if duration in duration_checker
    #                     else True,
    #                     legendgroup=duration,
    #                 ),
    #                 # row=intensity_count+1, col=1
    #                 row=1,
    #                 col=len(intensities) - intensity_count,
    #             )
    #             if len(y_toplot) != 0:
    #                 duration_checker.append(duration)

    #     # below is code from stack overflow to hide duplicate legends
    #     # names = set()
    #     # mean_traces_fig.for_each_trace(
    #     #     lambda trace:
    #     #         trace.update(showlegend=False)
    #     #         if (trace.name in names) else names.add(trace.name))

    #     mean_traces_fig.update_layout(
    #         title_text=self.dataset
    #         + ", "
    #         + self.cell_name
    #         + ", "
    #         + self.genotype,
    #         title_x=0.5,
    #         legend_title_text="Light Duration",
    #     )

    #     # mean_traces_fig.show()

    #     self.mean_traces_fig = mean_traces_fig

    # def output_html_plots(self):
    #     """
    #     Saves the sweep stats and mean trace plots as one html file
    #     """

    #     base_path = os.path.join(
    #         "/home/jhuang/Documents/phd_projects/MMZ_STC_dataset/figures",
    #         self.dataset,
    #         self.genotype,
    #     )
    #     if not os.path.exists(base_path):
    #         os.makedirs(base_path)

    #     html_filename = "{}_summary_plots.html".format(self.cell_name)
    #     path = os.path.join(base_path, html_filename)

    #     self.mean_traces_fig.write_html(
    #         path, full_html=False, include_plotlyjs="cdn"
    #     )

    #     # pdb.set_trace()
    #     # only append curve stats if cell has a response
    #     if self.response == True:
    #         with open(path, "a") as f:
    #             f.write(
    #                 self.curve_stats_fig.to_html(
    #                     full_html=False, include_plotlyjs=False
    #                 )
    #             )

