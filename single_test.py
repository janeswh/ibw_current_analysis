from cProfile import run
from pickle import FALSE
from urllib import response
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

import collections

from scipy.stats import sem
from scipy import io

import pymc3 as pm
from theano import shared

import plotly.io as pio

pio.renderers.default = "browser"

import p2_acq_parameters
import p14_acq_parameters
import plotting

from file_settings import FileSettings
import pdb


def igor_to_pandas(path_to_file):
    """This function opens an igor binary file (.ibw), extracts the time
    series data, and returns a pandas DataFrame"""

    data_raw = IgorIO(filename=path_to_file)
    data_neo = data_raw.read_block()
    data_neo_array = data_neo.segments[0].analogsignals[0]
    data_df = pd.DataFrame(data_neo_array.as_array().squeeze())

    return data_df


def run_BARS_smoothing(x_stop, x_array, y_array, x_plot):
    """
        Runs Bayesian Adaptive Regression Splines (BARS) smoothing, code
        adapted from 
        https://gist.github.com/AustinRochford/d640a240af12f6869a7b9b592485ca15
        but changed for updated pymc.

        See also:
        Wallstrom et al. 2008
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2748880/

        Statistical Smoothing of Neuronal Data by Rob Kass 
        https://www.stat.cmu.edu/~kass/papers/smooth.pdf

        Arguments:
            x (array): the x-values of the dataset, from the bar_bins of psth
            y (array): the y-values of the dataset, freq from psth
            x_plot (array): the x values to span the plot
        
        Returns:
            smoothed_avg (array): The average interpolated y-values
        """
    x = x_array[:x_stop]
    y = y_array[:x_stop]
    N_KNOT = 20  # arbitrary # of knots
    # quantiles = np.linspace(
    #     0, 1, N_KNOT
    # )  # if I want to use quantiles as knots
    knots = np.linspace(x_plot[0], x_plot[-1], N_KNOT)  # interior knots

    # feed interior knots to get spline coefficients
    knots, c, k = scipy.interpolate.splrep(x=x, y=y, task=-1, t=knots[1:-1])

    spline = scipy.interpolate.BSpline(knots, c, 3, extrapolate=False)

    BARS_fig = go.Figure()

    BARS_fig.add_trace(
        go.Scatter(x=x_plot, y=spline(x_plot), name="spline")
    )  # plots splines
    BARS_fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#D39DDD"),
            name="freq",
        )
    )  # plots freq points

    N_MODEL_KNOTS = 5 * N_KNOT
    # N_MODEL_KNOTS = N_KNOT
    model_knots = np.linspace(x_plot[0], x_plot[-1], N_MODEL_KNOTS)

    # running model
    basis_funcs = scipy.interpolate.BSpline(
        model_knots,  # why doesn't example code use model_knots here?
        np.eye(N_MODEL_KNOTS),
        k=3,
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

    smoothed_avg = pp_trace["obs"].mean(axis=0)

    BARS_fig.add_trace(
        go.Scatter(x=x_plot, y=smoothed_avg, name="spline estimate",)
    )
    # BARS_fig.show()

    return smoothed_avg


def decay_func(time, a, tau, offset):
    """
    Exponential decay function for calculating tau
    Parameters
    ----------
    time: array
        x array of time
    current_peak: scalar/float
        value of the starting peak for the exponential decay
    tau: scalar/float
        the time constant of decay of the exponential
    offset: scalar/float
        the asymptote that the exponential will decay to

    Returns
    -------
    y: array
        the y values for the exponential decay
    """
    return a * np.exp(-time / tau) + offset


def ipsc_func(time, peak, peak_time, tau):

    return peak * time * np.exp(-(time - peak_time) / tau)


class JaneCell(object):
    def __init__(self, dataset, sweep_info, file, file_name):
        self.dataset = dataset
        self.sweep_info = sweep_info
        self.file = file
        self.file_name = file_name
        self.drop_ibw = None
        self.time = None
        self.raw_sweep_length = None
        self.sweep_length_ms = None
        self.num_sweeps = None
        self.raw_df = None
        self.raw_ic_df = None
        self.spikes_sweeps_dict = None
        self.drug_sweeps_dict = None
        self.ic_sweeps_dict = None
        self.sweeps_dict = None
        self.mod_events_df = None
        self.event_stats = None
        self.updated_mod_events = None
        self.events_fig = None
        self.bar_bins = None

        self.mean_trace_fig = None

        self.freq = None
        self.avg_frequency_df = None
        self.avg_frequency_stats = None
        self.frequency_decay_fit = None

        self.decay_fits_dict = None
        self.event_stats_fig = None
        self.annotated_events_fig = None

        self.traces_filtered = None
        self.traces_filtered_sub = None
        self.sub_mean_trace = None
        self.mean_trace_stats = None
        self.cell_analysis_dict = None
        self.power_curve_df = None
        self.tuples = None
        self.power_curve_stats = None
        self.sweep_analysis_values = None
        self.cell_name = None
        self.cell_type = None
        self.condition = None
        self.response = None
        self.threshold = None

        # set acquisition parameters
        if self.dataset == "p2":
            self.lowpass_freq = p2_acq_parameters.LOWPASS_FREQ
            self.stim_time = p2_acq_parameters.STIM_TIME
            self.post_stim = p2_acq_parameters.POST_STIM
            self.freq_post_stim = p2_acq_parameters.FREQ_POST_STIM
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
            self.freq_post_stim = p14_acq_parameters.FREQ_POST_STIM
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
        self.raw_sweep_length = len(self.raw_df)
        self.sweep_length_ms = self.raw_sweep_length / self.fs
        self.traces = self.raw_df
        self.num_sweeps = len(self.raw_df.columns)

        # gets sweep info for one cell, drops empty values
        # file_split = self.file_name.split(".")
        file_split = self.file_name.split("_")
        self.cell_name = f"{file_split[0]}_{file_split[1]}"

        drop_file_split = self.file_name.split(".")
        self.drop_ibw = drop_file_split[0]  # used in get mod events

        cell_sweep_info = self.sweep_info.loc[
            self.sweep_info["File Path"] == self.file_name
        ]

        self.cell_type = cell_sweep_info["Cell Type"].values[0]

        if "light" in self.file_name:
            self.condition = "light"
        elif "spontaneous" in self.file_name:
            self.condition = "spontaneous"

        self.threshold = cell_sweep_info["Thresholding"].values[0]

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
        self, data, pos, baseline, window_width, polarity="-", index=False
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

        # shorten window width if pos occurs too early
        if pos < 50:
            window_width = 20

        start = int(pos - window_width / 2)
        end = int(pos + window_width / 2)

        peak_window = subtracted_data.iloc[start:end]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=peak_window.index, y=peak_window))
        # fig.show()

        if index is True:
            if polarity == "-":
                epsc_peaks = peak_window.min()
                try:
                    epsc_peaks_index = peak_window.idxmin()
                except ValueError:
                    pdb.set_trace()
                    print(pos)
                    print(start)
                    print(end)
                    raise ValueError()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
                epsc_peaks_index = peak_window.idxmax()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks, epsc_peaks_index

        elif index is False:
            if polarity == "-":
                epsc_peaks = peak_window.min()
            elif polarity == "+":
                epsc_peaks = peak_window.max()
            else:
                raise ValueError("polarity must either be + or -")
            return epsc_peaks

    def calculate_charge(
        self, peak_time, peak, root_time, root, event_window, tau
    ):
        """
        Calculates the charge transferred of synaptic event by taking its
        integral from root to extrapolated root (is this sketch?).

        Or integrate using the decay func?
        """
        # subtract root from trace
        event_window = event_window - root
        integral_start = root_time

        # arbitrarily find end time within 5 ms?

        event_fig = go.Figure()
        event_fig.add_trace(
            go.Scatter(x=event_window.index, y=event_window, name="event")
        )

        # event_fig.add_trace(
        #     go.Scatter(x=decay_fit["x"], y=decay_fit["y"], name="decay")
        # )

        # event_fig.show()
        # normalize

        x = event_window.index
        y = event_window
        norm_x = event_window.index.min()
        norm_y = event_window.min()

        x_2 = x - norm_x + 1  # why +1 here? so values don't start at 0
        y_2 = y / norm_y
        peak_time_norm = peak_time - norm_x + 1
        peak_norm = peak / norm_y
        # ys = scipy.integrate.odeint(ipsc_func, peak_time, time)

        # this is trying to use decay function
        y_func = peak_norm * x_2 * np.exp(-(x_2 - peak_time_norm) / tau)

        # event_fig.add_trace(go.Scatter(x=x_2, y=y_2, name="event"))

        # event_fig.add_trace(go.Scatter(x=x_2, y=y_func, name="ipsc fit"),)
        event_fig.show()

    def calculate_event_kinetics(self, data, peak, pos, baseline):
        """
        Gets kinetic stats of events
        """
        # in ms
        before_window = 5
        after_window = 20
        subtracted_data = data - baseline

        start = int(pos - before_window)
        end = int(pos + after_window)

        # using .loc here because indices are in ms
        event_window = subtracted_data.loc[start:end]

        # # calculate basline, try: going back 2 ms from peak, then 2 more ms
        # # for baseline
        peak_time = subtracted_data[subtracted_data == peak].index[0]

        # calculate root point for baseline
        root_start = peak_time - 5
        root_window = event_window[root_start:peak_time]
        root_time = root_window.idxmax()
        root = root_window.max()

        adjusted_peak = peak - root

        # tau, decay_fit = self.calculate_decay(
        #     peak, pos, event_window, data_type="event", polarity="-"
        # )

        # self.calculate_charge(
        #     peak_time, peak, root_time, root, event_window, tau
        # )

        rise_time, rise_start, rise_end = self.calculate_rise_time(
            peak,
            peak_time,
            event_window,
            polarity="-",
            root_time=root_time,
            data_type="event",
        )
        tau = None
        decay_fit = None
        return (
            tau,
            decay_fit,
            rise_time,
            rise_start,
            rise_end,
            adjusted_peak,
            root_time,
        )

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

    def calculate_responses(self, baseline_std, peak_mean, threshold=None):
        """
            Decides on whether there is a response above 2x, 3x above the baseline std,
            or a user-selectable cutoff.
            Parameters
            ----------
            baseline_std: int or float
                The std of the baseline of the mean filtered trace.
            peak_mean: int or float
                The current peak of the mean filtered trace.
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

    def drop_short_isi(self):
        # makes list of event times for each sweep and puts in df
        isi_df = pd.DataFrame()
        pos_drops = []
        for sweep in self.mod_events_df["Sweep"].unique():
            # sweep = 2
            pos = self.mod_events_df.loc[self.mod_events_df["Sweep"] == sweep][
                "Event pos (ms)"
            ]
            sweep_isi = pd.DataFrame(elephant.statistics.isi(pos))

            # finds events occuring less than 5 ms after the previous event
            to_discard = np.where(sweep_isi < 5)[0]
            pos_to_discard = to_discard + 1  # because event is shifted by one
            pos_discard_idx = pos.iloc[pos_to_discard].index
            pos_drops.append(pos_discard_idx.to_list())

            isi_df = pd.concat([isi_df, sweep_isi], axis=1)

        # drops short isi events
        # pos_drops contain absolute indices (not time) of events that can
        # be dropped from self.mod_events_df
        pos_drops_flat = [idx for sublist in pos_drops for idx in sublist]
        self.mod_events_df.drop(pos_drops_flat, inplace=True)

    def calculate_event_stats(self):
        # truncate traces to start after tp end
        # traces = self.traces[self.tp_start + self.tp_length :]

        traces = self.traces
        self.traces_filtered = self.filter_traces(traces)

        self.time = np.arange(0, len(traces) / self.fs, 1 / self.fs)
        self.traces_filtered.index = self.time

        # find baseline, defined as the last 3s of the sweep
        baseline = self.calculate_mean_baseline(self.traces_filtered)
        std_baseline = self.calculate_std_baseline(self.traces_filtered)

        # subtract mean baseline from all filtered traces - this is for
        self.traces_filtered_sub = self.traces_filtered - baseline

        # finds the new positions and peaks of identified events using MOD file
        all_new_amps = []  # for updated mod_events_df
        all_new_pos = []  # for updated mod_events_df

        sweep_number_list = []
        new_amps = []
        new_pos_list = []
        tau_list = []
        risetime_list = []
        risestart_list = []
        riseend_list = []
        adjusted_peak_list = []
        roots_list = []

        # drops positions with ISI > 5
        # self.drop_short_isi()

        decay_fits_dict = collections.defaultdict(dict)
        true_index = 0  # for dict key values

        for index, row in self.mod_events_df.iterrows():
            sweep = int(row["Sweep"])
            pos = int(row["Subtracted pos"])
            sweep_baseline = baseline[sweep]

            # finds event peak and new position
            event_peak, new_pos = self.calculate_event_peak(
                self.traces_filtered[sweep],
                pos,
                sweep_baseline,
                window_width=100,
                index=True,
            )

            # sets new event window to do kinetics analyses
            (
                tau,
                decay_fit,
                rise_time,
                rise_start,
                rise_end,
                adjusted_peak,
                root_time,
            ) = self.calculate_event_kinetics(
                self.traces_filtered[sweep],
                event_peak,
                new_pos,
                sweep_baseline,
            )

            # don't add decay fits to dict if tau > 100
            # if tau < 200:
            # decay_fits_dict[sweep][true_index] = decay_fit

            sweep_number_list.append(sweep)
            new_amps.append(event_peak)
            new_pos_list.append(new_pos)
            # tau_list.append(tau)
            risetime_list.append(rise_time)
            risestart_list.append(rise_start)
            riseend_list.append(rise_end)
            adjusted_peak_list.append(adjusted_peak)
            roots_list.append(root_time)

            true_index += (
                1  # makes it so dict keys match up with event_stats indices
            )

            all_new_amps.append(event_peak)
            all_new_pos.append(new_pos)

        new_stats = pd.DataFrame(
            {"New amplitude (pA)": all_new_amps, "New pos": all_new_pos}
        )
        self.updated_mod_events = pd.concat(
            [self.mod_events_df, new_stats], axis=1
        )

        # self.plot_mod_events()

        event_stats = pd.DataFrame(
            {
                "Sweep": sweep_number_list,
                "New amplitude (pA)": new_amps,
                "New pos": new_pos_list,
                # "Tau (ms)": tau_list,
                "Rise time (ms)": risetime_list,
                "Rise start (ms)": risestart_list,
                "Rise end (ms)": riseend_list,
                "Adjusted amplitude (pA)": adjusted_peak_list,
                "Root time (ms)": roots_list,
            }
        )

        # calculate decay fits afterwards so it has access to roots time
        # of the following event
        for index, row in event_stats.iterrows():

            sweep = int(row["Sweep"])
            pos = row["New pos"]
            amplitude = row["New amplitude (pA)"]
            sweep_baseline = baseline[sweep]

            if (
                index
                != event_stats.loc[event_stats["Sweep"] == sweep].index[-1]
            ):
                next_root = event_stats.loc[event_stats["Sweep"] == sweep][
                    "Root time (ms)"
                ][index + 1]
            # else:
            #     print("last event stop")
            #     pdb.set_trace()
            #     next_root = np.nan

            before_window = 5
            after_window = 20
            subtracted_data = self.traces_filtered[sweep] - sweep_baseline

            start = int(pos - before_window)
            end = int(pos + after_window)

            # using .loc here because indices are in ms
            event_window = subtracted_data.loc[start:end]

            # print(f"running decay fits for sweep {sweep} pos {pos}")

            tau, decay_fit = self.calculate_decay(
                amplitude,
                pos,
                event_window,
                data_type="event",
                polarity="-",
                next_root=next_root,
            )
            tau_list.append(tau)

            decay_fits_dict[sweep][index] = decay_fit

        tau_list = pd.DataFrame({"Tau (ms)": tau_list})

        event_stats = pd.concat([event_stats, tau_list], axis=1)

        # drop events with tau greater than 100
        # to_drop = event_stats.loc[event_stats["Tau (ms)"] > 100].index
        # event_stats.drop(to_drop, inplace=True)

        self.decay_fits_dict = decay_fits_dict
        self.event_stats = event_stats

    def plot_annotated_events(self):
        """
        Plots all the events for every sweep, with the decay fits, rise times,
        etc. This doesn't include eliminated events.
        """

        annotated_events_fig = go.Figure()

        # add main title
        annotated_events_fig.update_layout(
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} "
                f"annotated events"
            ),
            title_x=0.5,
        )

        for sweep in range(self.num_sweeps):

            # need to only plot events with rise times after the start of
            # window_toplot

            sweep_events = self.event_stats.loc[
                (self.event_stats["Sweep"] == sweep)
                & (
                    self.event_stats["Root time (ms)"]
                    > (self.tp_start + self.tp_length)
                )
            ]

            window_toplot = self.traces_filtered_sub[sweep][
                (self.tp_start + self.tp_length) : :
            ]

            sweep_events_pos = sweep_events.loc[
                sweep_events["Sweep"] == sweep
            ]["New pos"].values
            sweep_events_amp = sweep_events.loc[
                sweep_events["Sweep"] == sweep
            ]["New amplitude (pA)"].values

            sweep_events_tau = sweep_events.loc[
                sweep_events["Sweep"] == sweep
            ]["Tau (ms)"]

            # gets rise start info
            rise_starts_pos = sweep_events.loc[sweep_events["Sweep"] == sweep][
                "Rise start (ms)"
            ].values
            try:
                rise_starts_amp = window_toplot.loc[rise_starts_pos]
            except KeyError:
                pdb.set_trace()

            # gets rise end info
            rise_ends_pos = sweep_events.loc[sweep_events["Sweep"] == sweep][
                "Rise end (ms)"
            ].values
            rise_ends_amp = window_toplot.loc[rise_ends_pos]

            # gets roots for baseline
            roots_pos = sweep_events.loc[sweep_events["Sweep"] == sweep][
                "Root time (ms)"
            ].values
            try:
                roots_amp = window_toplot.loc[roots_pos]
            except KeyError:
                pdb.set_trace()

            # gets root-subtracted amplitudes for annotating
            adj_amplitudes = sweep_events.loc[sweep_events["Sweep"] == sweep][
                "Adjusted amplitude (pA)"
            ]

            # plots the sweep
            annotated_events_fig.add_trace(
                go.Scatter(
                    x=window_toplot.index,
                    y=window_toplot,
                    mode="lines",
                    name="sweep {}".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            # plots peaks and tau
            annotated_events_fig.add_trace(
                go.Scatter(
                    x=sweep_events_pos,
                    y=sweep_events_amp,
                    mode="markers + text",
                    marker=dict(color="#E75649", size=12),
                    # text=f"{adj_amplitudes.round(2)}, tau={sweep_events_tau.round(2)}",
                    text=adj_amplitudes.round(2),
                    textfont=dict(size=18),
                    textposition="bottom center",
                    name="sweep {} peaks".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            # plots start end
            annotated_events_fig.add_trace(
                go.Scatter(
                    x=rise_starts_pos,
                    y=rise_starts_amp,
                    mode="markers",
                    marker=dict(color="#8FE749", size=12),
                    name="sweep {} rise start".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            # plots rise end
            annotated_events_fig.add_trace(
                go.Scatter(
                    x=rise_ends_pos,
                    y=rise_ends_amp,
                    mode="markers",
                    marker=dict(color="#3E5BCF", size=12),
                    name="sweep {} rise end".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            # plots roots
            annotated_events_fig.add_trace(
                go.Scatter(
                    x=roots_pos,
                    y=roots_amp,
                    mode="markers +text",
                    marker=dict(color="#993ECF", size=8),
                    text=sweep_events_tau.round(2),
                    name="sweep {} roots/baseline".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            # plots decay fits
            # only plot fits for events that are also in sweep_events
            # skips event with nan value for tau/no decay fit
            for event in sweep_events.index:
                if len(self.decay_fits_dict[sweep][event]) != 0:
                    annotated_events_fig.add_trace(
                        go.Scatter(
                            x=self.decay_fits_dict[sweep][event]["x"],
                            y=self.decay_fits_dict[sweep][event]["y"],
                            line=dict(color="#E89F24"),
                            mode="lines",
                            name="sweep {} decay fits".format(sweep),
                            legendgroup=sweep,
                            visible="legendonly",
                        )
                    )

        # below is code from stack overflow to hide duplicate legends
        names = set()
        annotated_events_fig.for_each_trace(
            lambda trace: trace.update(showlegend=False)
            if (trace.name in names)
            else names.add(trace.name)
        )

        annotated_events_fig.update_xaxes(title_text="Time (ms)")
        annotated_events_fig.update_yaxes(title_text="Amplitude (pA)")

        annotated_events_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        # annotated_events_fig.show()

        self.annotated_events_fig = annotated_events_fig

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

        # find peak time
        mean_trace_peak_time = (
            self.sub_mean_trace[0]
            .loc[self.sub_mean_trace[0] == mean_trace_peak[0]]
            .index[0]
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

        # determines whether the cell is responding, using mean_trace_filtered
        responses = self.calculate_responses(
            mean_std_baseline, mean_trace_peak
        )

        # collects measurements into cell dict, nested dict for each stim condition
        self.mean_trace_stats = pd.DataFrame(
            {
                "Cell name": self.cell_name,
                "Dataset": self.dataset,
                "Cell Type": self.cell_type,
                "Mean Trace Peak (pA)": mean_trace_peak[0],
                "Mean Trace Peak Time (ms)": mean_trace_peak_time,
                # "Mean Trace Onset Latency (ms)": mean_trace_latency[0],
                # "Mean Trace Time to Peak (ms)": mean_trace_time_to_peak[0],
                "Response 2x STD": responses["Response 2x STD"][0],
                "Response 3x STD": responses["Response 3x STD"][0],
            },
            index=[0],
        )

    def plot_mod_events(self):
        """
        Sanity check; plots all the individual sweeps with identified mod event 
        peaks marked. By default, all traces are hidden and only shown when
        selected in figure legends. This includes all events that are later
        dropped using kinetics criteria.
        """

        mod_events_fig = go.Figure()

        # add main title
        mod_events_fig.update_layout(
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} "
                f"all MOD events"
            ),
            title_x=0.5,
        )

        for sweep in range(self.num_sweeps):

            sweep_events_pos = self.updated_mod_events.loc[
                self.updated_mod_events["Sweep"] == sweep
            ]["New pos"].values
            sweep_events_amp = self.updated_mod_events.loc[
                self.updated_mod_events["Sweep"] == sweep
            ]["New amplitude (pA)"].values

            window_toplot = self.traces_filtered_sub[sweep][
                (self.tp_start + self.tp_length) : :
            ]

            mod_events_fig.add_trace(
                go.Scatter(
                    x=window_toplot.index,
                    y=window_toplot,
                    mode="lines",
                    name="sweep {}".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

            mod_events_fig.add_trace(
                go.Scatter(
                    x=sweep_events_pos,
                    y=sweep_events_amp,
                    mode="markers",
                    name="sweep {} peaks".format(sweep),
                    legendgroup=sweep,
                    visible="legendonly",
                )
            )

        mod_events_fig.update_xaxes(title_text="Time (ms)")
        mod_events_fig.update_yaxes(title_text="Amplitude (pA)")

        mod_events_fig.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        mod_events_fig.show()
        self.mod_events_fig = mod_events_fig

    def plot_events(self):
        """
        Sanity check; plots all the individual sweeps with identified event 
        peaks marked. By default, all traces are hidden and only shown when
        selected in figure legends.
        """
        events_fig = go.Figure()

        for sweep in range(self.num_sweeps):

            sweep_events_pos = self.event_stats.loc[
                self.event_stats["Sweep"] == sweep
            ]["New pos"].values
            sweep_events_amp = self.event_stats.loc[
                self.event_stats["Sweep"] == sweep
            ]["New amplitude (pA)"].values

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

    def analyze_avg_frequency(self, bin_width=10, time_stop=None):
        """
        bin_width is width of bins in ms
        time_stop is time of last timepoint wanted for BARS, in ms
        time_start is time when tp finishes

        1. Gets the histogram counts and parameters for PSTH
        2. Plots raster plot and PSTH for the entire sweep
        3. Smoothes PSTH using BARS and gets smoothed avg frequency
        4. Plots smoothed PSTH
        5. Does calculations on avg frequency 
        6. Plots smoothed PSTH with stats annotated if light condition
        
        """
        if time_stop is None:
            time_stop = int(self.sweep_length_ms)

        time_start = self.tp_start + self.tp_length

        raster_df, bins, bar_bins, freq = self.get_bin_parameters(
            bin_width, time_start,
        )

        # calculates bins for window
        (
            windowed_raster_df,
            windowed_bins,
            windowed_bar_bins,
            windowed_freq,
        ) = self.get_bin_parameters(bin_width, time_start, 2000)

        # trying to shorten BARS window to 1500 ms after stim time
        windowed_x_stop = len(windowed_bins)
        windowed_x_plot = np.linspace(
            windowed_bins[0], windowed_bins[-1], windowed_x_stop
        )

        x_stop = len(bins)  # number of bins to stop at

        self.plot_event_psth(raster_df, x=bar_bins, y=freq)

        x_plot = np.linspace(
            bins[0], bins[-1], x_stop
        )  # also time of avg_frequency

        # baseline subtract freq counts before smoothing
        baseline_window = self.stim_time - time_start
        baseline_bins = int(baseline_window / bin_width)
        # this stops before bins that would include cts from 500-510 ms
        baseline_freq = freq[:baseline_bins].mean()

        windowed_freq_sub = windowed_freq

        windowed_raw_avg_frequency = run_BARS_smoothing(
            windowed_x_stop,
            x_array=windowed_bar_bins,
            y_array=windowed_freq,
            x_plot=windowed_x_plot,
        )

        self.plot_smoothed_PSTH(
            windowed_x_stop,
            x_array=windowed_bar_bins,
            y_array=windowed_freq,
            x_plot=windowed_x_plot,
            smoothed=windowed_raw_avg_frequency,
        )

        self.calculate_avg_freq_stats(
            bin_width, windowed_x_plot, windowed_raw_avg_frequency
        )

        # this plots the BARS smoothing for response window ontop of freq
        # histogram for the whole sweep
        self.plot_annotated_freq_histogram(
            raster_df,
            windowed_x_stop,
            x_array=bar_bins,
            y_array=freq,
            x_plot=windowed_x_plot,
            smoothed=windowed_raw_avg_frequency,
        )

        # raw_avg_frequency = run_BARS_smoothing(
        #     x_stop, x_array=bar_bins, y_array=freq, x_plot=x_plot
        # )
        # self.plot_smoothed_PSTH(
        #     x_stop,
        #     x_array=bar_bins,
        #     y_array=freq,
        #     x_plot=x_plot,
        #     smoothed=raw_avg_frequency,
        # )

        # # check whether this is sketch
        # avg_frequency = self.replace_avg_extrapolation(freq, raw_avg_frequency)

        # self.calculate_avg_freq_stats(bin_width, x_plot, avg_frequency)

        # self.plot_annotated_freq_histogram(
        #     raster_df,
        #     x_stop,
        #     x_array=bar_bins,
        #     y_array=freq,
        #     x_plot=x_plot,
        #     smoothed=avg_frequency,
        # )

        self.freq = pd.DataFrame(freq, index=bar_bins)

    def plot_event_psth(self, event_times, x, y):
        """
        Makes raster plot of all identified events for each sweep.
        """
        # make sweep numbers go from 1-30 instead of 0-29
        new_sweeps = event_times["Sweep"] + 1

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
                x=event_times["New pos"],
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
            tickvals=[1, self.num_sweeps],
            showgrid=False,
            zeroline=False,
        )

        psth_fig.update_xaxes(
            row=1, col=1, showticklabels=False, showgrid=False
        )

        psth_fig.update_xaxes(
            # title_text="Time (ms)",
            range=[(self.tp_start + self.tp_length), self.sweep_length_ms],
        )

        psth_fig.add_trace(
            go.Bar(x=x, y=y, showlegend=False, marker=dict(color="#D39DDD"),),
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
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} PSTH"
            ),
            title_x=0.5,
        )

        # psth_fig.show()

    def get_bin_parameters(self, bin_width, time_start, time_stop=None):
        """
        Gets the counts, bins, bin widths for PSTH-related plotting and
        calculations. This uses the entire sweep.
        """
        # gets event positions for raster plot
        raster_df = self.event_stats[["Sweep", "New pos"]]
        # make PSTH
        psth_df = raster_df["New pos"]
        if time_stop == None:
            time_stop = self.sweep_length_ms
        bin_stop = int(time_stop) + bin_width

        counts, bins = np.histogram(
            psth_df, bins=range(time_start, bin_stop, bin_width)
        )
        # this puts bar in between the edges of the bin
        bar_bins = 0.5 * (bins[:-1] + bins[1:])

        # do frequency in Hz, # of events per second, divided by # of sweeps
        freq = counts / self.num_sweeps / 1e-2

        return raster_df, bins, bar_bins, freq

    def plot_smoothed_PSTH(self, x_stop, x_array, y_array, x_plot, smoothed):

        x = x_array[:x_stop]
        y = y_array[:x_stop]

        smoothed_psth = go.Figure()

        smoothed_psth.add_trace(
            go.Bar(x=x, y=y, marker=dict(color="#D39DDD"), name="PSTH",),
        )

        # this removes the white outline of the bar graph to emulate histogram
        smoothed_psth.update_traces(marker=dict(line=dict(width=0)),)

        smoothed_psth.update_yaxes(title_text="Frequency (Hz)")
        smoothed_psth.update_xaxes(title_text="Time (ms)")

        # add main title, x-axis titles
        smoothed_psth.update_layout(
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} PSTH"
            ),
            title_x=0.5,
        )
        smoothed_psth.update_layout(bargap=0)

        # adds blue overlay to show light stim duration
        smoothed_psth.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        smoothed_psth.add_trace(
            go.Scatter(
                x=x_plot,
                y=smoothed,
                marker=dict(color="#A613C4", size=2),
                name="spline estimate",
            )
        )

        # smoothed_psth.show()

    def replace_avg_extrapolation(self, freq, smoothed):
        # replace tail end of avg trace, if freq = 0, replace with pre-stim
        # baseline. is this sketchy?

        # finds last segment where there's no activity
        first_zero = np.where(freq > 0)[0][-1] + 1
        patch_length = len(freq[first_zero:])

        # replaces the interpolated frequency of last segment with start segment
        smoothed[first_zero:] = smoothed[: patch_length + 1]

        extrapolated_replaced = smoothed

        return smoothed

    def calculate_freq_baseline(
        self, baseline_start_idx, response_window_start, window="pre-stim"
    ):

        if window == "second half":
            # this is using latter half of sweep as baseline
            baseline_freq = self.avg_frequency_df.iloc[baseline_start_idx:]
        elif window == "pre-stim":
            # this is using pre-stim time as baseline
            baseline_freq = self.avg_frequency_df.iloc[:response_window_start]

        avg_baseline_freq = baseline_freq.mean()[0]
        std_baseline_freq = baseline_freq.std()

        return avg_baseline_freq, std_baseline_freq

    def calculate_freq_peak(self, freq_df):

        max_freq = freq_df.loc[self.stim_time :].max(axis=0)[0]
        freq_peak_time = freq_df.loc[500:].idxmax(axis=0)[0]
        time_to_peak_freq = freq_peak_time - self.stim_time

        return max_freq, freq_peak_time, time_to_peak_freq

    def calculate_freq_peak_onset(self, baseline_std, response_window):

        # how to define onset - 5% of peak, or exceeds 3 std above baseline
        # the issue is that the smoothed curve is already high before light stim

        # onset code probably won't work for spontaneous conditions since there
        # is no response

        onset_amp = baseline_std * 3
        onset_idx = np.argmax(response_window >= onset_amp)
        onset_time = response_window.index[onset_idx]

        return onset_time

    def calculate_rise_time(
        self,
        peak,
        peak_time,
        response_window,
        polarity,
        root_time=None,
        data_type=None,
    ):
        # do root subtraction for events
        if data_type == "event":
            response_window = response_window[root_time:peak_time]
            root = response_window[root_time]
            response_window = response_window - root
            peak = peak - root

        # rise time - 20-80%, ms
        if polarity == "-":
            rise_start_idx = np.argmax(response_window <= peak * 0.2)
            rise_end_idx = np.argmax(response_window <= peak * 0.8)
        elif polarity == "+":
            rise_start_idx = np.argmax(response_window >= peak * 0.2)
            rise_end_idx = np.argmax(response_window >= peak * 0.8)

        rise_start = response_window.index[rise_start_idx]
        rise_end = response_window.index[rise_end_idx]

        rise_time = rise_end - rise_start

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=response_window.index, y=response_window, name="event"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[rise_start],
                y=[response_window.loc[rise_start]],
                name="rise start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[rise_end],
                y=[response_window.loc[rise_end]],
                name="rise end",
            )
        )
        # fig.show()

        return rise_time, rise_start, rise_end

    def define_decay_parameters(
        self,
        max_freq,
        freq_peak_time,
        response_window,
        data_type,
        polarity,
        freq_baseline=None,
        next_root=None,
        retry=False,
    ):
        """
        Defines window passed into decay fit
        Tries to find decay end point using:
        1. 10% of peak, within 20 ms after peak time
        2. If value doesn't exist within 20 ms, use peak time + 3 ms
        3. If tau from above end time is too large, use next root as end time
        4. If next root is more than 20 ms after peak time, give up
        """
        # decay window is peak to 90% decay, overall window is peak to 20 ms
        # after peak for events
        if data_type == "event":
            decay_array = response_window.loc[
                freq_peak_time : freq_peak_time + 20
            ]

        # baseline subtract if it's frequency
        if data_type == "frequency":
            decay_array = response_window.loc[freq_peak_time:]
            decay_array = decay_array - freq_baseline
            max_freq = decay_array.iloc[0]

        if polarity == "-":
            decay_end_idx = np.argmax(decay_array >= max_freq * 0.1)
        elif polarity == "+":
            decay_end_idx = np.argmax(decay_array <= max_freq * 0.1)

        # if the 90% decay isn't found within the decay window, use time point
        # that is 3 ms after peak if event, and first time it returns to
        # baseline if frequency

        if decay_end_idx == 0:
            if data_type == "event":
                decay_end_time = freq_peak_time + 3
            elif data_type == "frequency":
                decay_end_time = decay_array.index[-1]
        else:
            decay_end_time = decay_array.index[decay_end_idx]

        # if tau is huge, have the decay end point be the next root pos
        if data_type == "event" and retry is True:
            decay_end_time = next_root

            # if revised decay end time occurs 20 ms after event, give up on this
            # event and return empty decay window
            if decay_end_time > freq_peak_time + 20:
                decay_window = pd.Series([])
            else:
                decay_window = response_window.loc[
                    freq_peak_time:decay_end_time
                ]

        else:
            decay_window = response_window.loc[freq_peak_time:decay_end_time]

        if polarity == "-":
            decay_window = decay_window * -1
            max_freq = max_freq * -1

        return decay_window

    def do_decay_fit(self, x_2, y_2):
        """
        Normalize x and y data and do decay fit
        """

        starting_params = [1, 1, 1]

        # fits
        try:
            popt, pcov = scipy.optimize.curve_fit(
                f=decay_func,
                # xdata=decay_window.index.to_numpy(),
                xdata=x_2,
                ydata=y_2,
                p0=starting_params,
                bounds=((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf)),
            )

        except RuntimeError:
            popt = (np.nan, np.nan, np.nan)

        except ValueError as exc:
            pdb.set_trace()
            print(exc)

        return popt

    def normalize_arrays(self, decay_window, x, y):
        """
        Normalizes x and y arrays for fitting decay exponential
        """

        norm_y = decay_window.min()
        y_plot = decay_window

        # normalize
        norm_x = decay_window.index.min()

        x_2 = x - norm_x + 1  # why +1 here? so values don't start at 0
        y_2 = y / norm_y

        return x_2, y_2, y_plot, norm_y

    def calculate_decay(
        self,
        max_freq,
        freq_peak_time,
        response_window,
        data_type,
        polarity,
        freq_baseline=None,
        next_root=None,
    ):

        decay_window = self.define_decay_parameters(
            max_freq,
            freq_peak_time,
            response_window,
            data_type,
            polarity,
            freq_baseline=freq_baseline,
        )

        x = decay_window.index.to_numpy()
        y = decay_window.to_numpy()

        x_2, y_2, y_plot, norm_y = self.normalize_arrays(decay_window, x, y)
        popt = self.do_decay_fit(x_2, y_2)
        a, tau, offset = popt

        if tau > 100 and data_type == "event":
            # print(f"retrying fits for pos: {freq_peak_time}")
            decay_window = self.define_decay_parameters(
                max_freq,
                freq_peak_time,
                response_window,
                data_type,
                polarity,
                next_root=next_root,
                retry=True,
            )

            x = decay_window.index.to_numpy()
            y = decay_window.to_numpy()

            x_2, y_2, y_plot, norm_y = self.normalize_arrays(
                decay_window, x, y
            )
            # this is what happens when the next root is outside of response
            # window and then everything sucks and I give up on this tau
            if len(y) == 0:
                tau = np.nan
                decay_fit = np.nan
            else:
                popt = self.do_decay_fit(x_2, y_2)
                a, tau, offset = popt

        # if tau is STILL huge, make it nan
        if tau > 100:
            tau = np.nan

        if len(y) != 0 and tau != np.nan:
            decay_fit = decay_func(x_2, *popt)

            decay_fig = go.Figure()
            decay_fig.add_trace(go.Scatter(x=x, y=y_plot, name="data",))
            decay_fig.add_trace(
                go.Scatter(x=x, y=norm_y * decay_fit, name="fit on 90%",)
            )
            decay_fig.add_trace(
                go.Scatter(
                    x=response_window.index,
                    y=response_window * -1,
                    name="event",
                )
            )

        # un-normalize y-values before returning

        if polarity == "-":
            decay_fit_trace = -1 * decay_fit * norm_y
        elif polarity == "+":
            decay_fit_trace = decay_fit * norm_y

        decay_fit = pd.DataFrame({"x": x, "y": decay_fit_trace})

        return tau, decay_fit

    def calculate_avg_freq_stats(self, bin_width, x_plot, avg_frequency):

        avg_frequency_df = pd.DataFrame()
        avg_frequency_df["Avg Frequency (Hz)"] = avg_frequency
        avg_frequency_df.index = x_plot
        self.avg_frequency_df = avg_frequency_df

        baseline_start_idx = int(self.baseline_start / bin_width)

        # window to look for response starts after light stim
        if self.dataset == "p2":
            response_window_start = int(self.stim_time / bin_width)
            response_window_end = int(self.freq_post_stim / bin_width)

            response_window = self.avg_frequency_df.iloc[
                response_window_start:response_window_end
            ]
        elif self.dataset == "p14":
            response_window_start = self.stim_time
            response_window_end = self.freq_post_stim

            response_window = self.avg_frequency_df.loc[
                response_window_start:response_window_end
            ]

        avg_baseline_freq, std_baseline_freq = self.calculate_freq_baseline(
            baseline_start_idx, response_window_start,
        )

        (
            max_freq,
            freq_peak_time,
            time_to_peak_freq,
        ) = self.calculate_freq_peak(avg_frequency_df)

        onset_time = self.calculate_freq_peak_onset(
            std_baseline_freq, response_window
        )

        if self.condition == "spontaneous":
            rise_time, rise_start, rise_end, tau, decay_fit = (None,) * 5
            self.response = False

        elif self.condition == "light":
            if (max_freq > 1) and (freq_peak_time < 1500):
                rise_time, rise_start, rise_end = self.calculate_rise_time(
                    max_freq,
                    freq_peak_time,
                    response_window,
                    polarity="+",
                    data_type="frequency",
                )

                tau, decay_fit = self.calculate_decay(
                    max_freq,
                    freq_peak_time,
                    response_window["Avg Frequency (Hz)"],
                    data_type="frequency",
                    polarity="+",
                    freq_baseline=avg_baseline_freq,
                )
                self.response = True
            else:
                rise_time, rise_start, rise_end, tau, decay_fit = (None,) * 5
                self.response = False

        avg_freq_stats = pd.DataFrame(
            {
                "Peak Frequency (Hz)": max_freq,
                "Peak Frequency Time (ms)": freq_peak_time,
                "Time to Peak Frequency (ms)": time_to_peak_freq,
                "Baseline Frequency (Hz)": avg_baseline_freq,
                "Baseline-sub Peak Freq (Hz)": max_freq - avg_baseline_freq,
                "Response Onset Latency (ms)": onset_time - self.stim_time,
                "Rise Time (ms)": rise_time,
                "Decay tau": tau,
            },
            index=[0],
        )

        self.frequency_decay_fit = decay_fit
        self.avg_frequency_stats = avg_freq_stats

    def plot_annotated_freq_histogram(
        self, event_times, x_stop, x_array, y_array, x_plot, smoothed
    ):
        """
        Makes raster plot + annotated freq histogram, with decay fit if
        self.response is True

        Then adds mean trace below, shared axes, with peak annotated
        """

        # make sweep numbers go from 1-30 instead of 0-29
        new_sweeps = event_times["Sweep"] + 1

        # sets background color to white
        layout = go.Layout(plot_bgcolor="rgba(0,0,0,0)",)

        # make overall fig layout
        annotated_freq = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.025,
            x_title="Time (ms)",
            shared_xaxes=True,
        )

        # add raster plot
        annotated_freq.add_trace(
            go.Scatter(
                x=event_times["New pos"],
                y=new_sweeps,
                mode="markers",
                marker=dict(symbol="line-ns", line_width=1, size=10),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        annotated_freq.update_yaxes(
            title_text="Trial",
            row=1,
            col=1,
            tickvals=[1, self.num_sweeps],
            showgrid=False,
            zeroline=False,
        )

        annotated_freq.update_xaxes(
            row=1, col=1, showticklabels=False, showgrid=False
        )

        # x = x_array[:x_stop]
        # y = y_array[:x_stop]

        # annotated_freq = go.Figure()

        # this plots the event histogram for the entire sweep
        annotated_freq.add_trace(
            go.Bar(
                x=x_array,
                y=y_array,
                marker=dict(color="#D39DDD"),
                name="PSTH",
            ),
            row=2,
            col=1,
        )

        # this removes the white outline of the bar graph to emulate histogram
        annotated_freq.update_traces(
            marker=dict(line=dict(width=0)), row=2, col=1,
        )

        annotated_freq.update_yaxes(
            title_text="Frequency (Hz)", row=2, col=1,
        )

        # add main title, x-axis titles
        annotated_freq.update_layout(
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} PSTH"
            ),
            title_x=0.5,
        )

        annotated_freq.update_layout(bargap=0)

        # adds blue overlay to show light stim duration
        annotated_freq.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
            row="all",
            col=1,
        )

        annotated_freq.add_trace(
            go.Scatter(
                x=x_plot,
                y=smoothed,
                marker=dict(color="#A613C4", size=2),
                name="spline estimate",
            ),
            row=2,
            col=1,
        )

        if self.response is True:
            # add decay fit
            annotated_freq.add_trace(
                go.Scatter(
                    x=self.frequency_decay_fit["x"],
                    y=self.frequency_decay_fit["y"],
                    marker=dict(color="#A4F258", size=2),
                    name="decay fit",
                ),
                row=2,
                col=1,
            )

        # add peak frequency
        annotated_freq.add_trace(
            go.Scatter(
                x=[self.avg_frequency_stats["Peak Frequency Time (ms)"][0]],
                y=[self.avg_frequency_stats["Peak Frequency (Hz)"][0]],
                marker=dict(color="#FFB233", size=4),
                name="peak frequency",
            ),
            row=2,
            col=1,
        )

        # plots mean trace with annotations
        window_toplot = self.sub_mean_trace[
            (self.tp_start + self.tp_length) : :
        ]
        annotated_freq.add_trace(
            go.Scatter(
                x=window_toplot.index,
                y=window_toplot.squeeze(),
                mode="lines",
                marker=dict(color="#5EC320", size=2),
                name="averaged sweep",
                # legendgroup="avg_sweep",
                # visible="legendonly",
            ),
            row=3,
            col=1,
        )

        # add mean trace peak
        annotated_freq.add_trace(
            go.Scatter(
                x=[self.mean_trace_stats["Mean Trace Peak Time (ms)"][0]],
                y=[self.mean_trace_stats["Mean Trace Peak (pA)"][0]],
                marker=dict(color="#3338FF", size=4),
                name="mean trace peak",
            ),
            row=3,
            col=1,
        )

        annotated_freq.add_vrect(
            x0=self.stim_time,
            x1=self.stim_time + 100,
            fillcolor="#33F7FF",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        self.annotated_freq_fig = annotated_freq
        # annotated_freq.show()

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

    def plot_mean_trace(self):
        """
        Plots the averaged trace ontop of all individual sweeps. Individual 
        sweeps are hidden by default unless selected to show with legend.
        """

        mean_trace_fig = go.Figure()

        # add main title
        mean_trace_fig.update_layout(
            title_text=(
                f"{self.cell_name}, {self.cell_type}, {self.condition} "
                f"mean trace"
            ),
            title_x=0.5,
        )

        # # plots individual sweeps
        # for sweep in range(self.num_sweeps):
        #     indiv_toplot = self.traces_filtered_sub[sweep][
        #         (self.tp_start + self.tp_length) : :
        #     ]

        #     mean_trace_fig.add_trace(
        #         go.Scatter(
        #             x=indiv_toplot.index,
        #             y=indiv_toplot,
        #             mode="lines",
        #             marker=dict(color="gray", line_width=1),
        #             name="sweep {}".format(sweep),
        #             legendgroup="indiv_sweeps",
        #             visible="legendonly",
        #         )
        #     )

        # plots averaged trace second so it shows up on top
        window_toplot = self.sub_mean_trace[
            (self.tp_start + self.tp_length) : :
        ]
        mean_trace_fig.add_trace(
            go.Scatter(
                x=window_toplot.index,
                y=window_toplot.squeeze(),
                mode="lines",
                # name="Averaged sweep",
                # legendgroup="avg_sweep",
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

        # mean_trace_fig.show()
        self.mean_trace_fig = mean_trace_fig

    def get_mod_events(self):

        mod_file = f"{self.drop_ibw}.mod.w4.e1.h13.minidet.mat"
        mod_events = io.loadmat(
            f"/home/jhuang/Documents/phd_projects/injected_GC_data/mod_events/"
            f"{self.dataset}/{mod_file}"
        )

        pos_list = mod_events["pos"]
        flat_pos = [pos for sublist in pos_list for pos in sublist]

        # get suprathreshold events for MOD files with threshold values
        if self.threshold == "Yes":
            suprathreshold_file = f"{self.drop_ibw}_suprathreshold.csv"
            suprathreshold_csv = pd.read_csv(
                f"/home/jhuang/Documents/phd_projects/scored_GC_data/original/"
                f"{self.dataset}/{self.cell_name}/{suprathreshold_file}"
            )
            suprathreshold_events_pos = suprathreshold_csv.loc[
                suprathreshold_csv["type"] == 1
            ]["position"].values.tolist()

            # add suprathreshold events into main pos list
            for suprathreshold_pos in suprathreshold_events_pos:
                flat_pos.append(suprathreshold_pos)

            # sort so all events are in order
            flat_pos.sort()

        flat_pos = pd.DataFrame(flat_pos)

        raw_intervals = np.arange(
            0,
            (self.num_sweeps + 1) * self.raw_sweep_length,
            self.raw_sweep_length,
        )

        # assign sweep numbers based on event positions
        sweep_pos = pd.cut(
            flat_pos[0],
            raw_intervals,
            labels=np.arange(0, self.num_sweeps),
            include_lowest=True,
            right=False,
        )

        mod_events_df = pd.DataFrame()
        mod_events_df["Sweep"] = sweep_pos
        mod_events_df["Raw pos"] = flat_pos

        sweep_array = np.asarray(mod_events_df["Sweep"])

        # get absolute time of event within each sweep
        subtract_time = sweep_array * self.fs * self.sweep_length_ms
        mod_events_df["Subtracted pos"] = flat_pos.squeeze() - subtract_time
        mod_events_df["Event pos (ms)"] = (
            mod_events_df["Subtracted pos"] / self.fs
        )

        # drop evnts occuring before tp ends
        # self.mod_events_df = mod_events_df.loc[
        #     mod_events_df["Event pos (ms)"] > self.tp_start + self.tp_length
        # ]
        self.mod_events_df = mod_events_df

    def make_cell_analysis_dict(self):

        # create dict to hold analysis results for each stim set
        filtered_traces_dict = {}
        cell_analysis_dict = {}
        power_curve_df = pd.DataFrame()
        all_mean_traces = pd.DataFrame()

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

    def save_mean_trace_plot(self):
        base_path = (
            f"{FileSettings.FIGURES_FOLDER}/{self.dataset}/{self.cell_type}/"
            f"{self.cell_name}"
        )

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        html_filename = f"{self.cell_name}_mean_traces.html"
        path = os.path.join(base_path, html_filename)

        self.mean_trace_fig.write_html(
            path, full_html=False, include_plotlyjs="cdn"
        )

    def save_annotated_events_plot(self):
        base_path = (
            f"{FileSettings.FIGURES_FOLDER}/{self.dataset}/{self.cell_type}/"
            f"{self.cell_name}"
        )

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        html_filename = f"{self.cell_name}_{self.condition}_events.html"
        path = os.path.join(base_path, html_filename)

        self.annotated_events_fig.write_html(
            path, full_html=False, include_plotlyjs="cdn"
        )

    def save_annotated_freq(self):
        """
        Saves raster plot with annotated freq histogram
        """
        base_path = (
            f"{FileSettings.FIGURES_FOLDER}/{self.dataset}/{self.cell_type}/"
            f"{self.cell_name}"
        )

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        html_filename = (
            f"{self.cell_name}_{self.condition}_annotated_freq.html"
        )
        path = os.path.join(base_path, html_filename)

        self.annotated_freq_fig.write_html(
            path, full_html=False, include_plotlyjs="cdn"
        )

