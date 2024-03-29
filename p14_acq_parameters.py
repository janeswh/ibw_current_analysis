""" ################### SET/CHECK THESE PARAMETERS BEFORE RUNNING ################## """
LOWPASS_FREQ = 500  # Hz
STIM_TIME = 500  # ms
POST_STIM = 150  # ms, amount of time after stimulus to look for freq and mean trace peak
RESPONSE_WINDOW_END = 2000  # ms, time response window ends
TP_START = 50  # ms, time of start of test pulse
TP_LENGTH = 300
VM_JUMP = -5  # mV, test pulse voltage jump
PRE_TP = 11  # ms, amount of time before test pulse start to get baseline
UNIT_SCALER = -12  # unitless, scaler to get back to A, from pA
AMP_FACTOR = 1000  # scaler for making plots in pA
FS = 10  # kHz, the sampling frequency
BASELINE_START = TP_START + TP_LENGTH  # starts when tp ends
BASELINE_END = STIM_TIME - 10  # ends 10 s before stim starts

