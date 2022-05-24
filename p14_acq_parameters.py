""" ################### SET/CHECK THESE PARAMETERS BEFORE RUNNING ################## """
LOWPASS_FREQ = 500  # Hz
STIM_TIME = 500  # ms
POST_STIM = (
    250  # ms, amount of time after stimulus to look for mean trace peak
)
FREQ_POST_STIM = 2000  # ms, amount of time after stim to look for max freq
TP_START = 50  # ms, time of start of test pulse
TP_LENGTH = 300
VM_JUMP = -5  # mV, test pulse voltage jump
PRE_TP = 11  # ms, amount of time before test pulse start to get baseline
UNIT_SCALER = -12  # unitless, scaler to get back to A, from pA
AMP_FACTOR = 1000  # scaler for making plots in pA
FS = 10  # kHz, the sampling frequency
BASELINE_START = TP_START + TP_LENGTH + 5  # 3000
BASELINE_END = STIM_TIME - 5

